import torch
from diffusers import StableDiffusionPipeline
from src.cfr_utils import *
from src.dataset import MACEDataset
import gc


def main(args):   
        
    model_id = f"{args.output_dir}"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lora_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    lora_pipe.safety_checker = None
    lora_pipe.requires_safety_checker = False
    
    final_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to("cuda")
    final_pipe.safety_checker = None
    final_pipe.requires_safety_checker = False
    final_projection_matrices, _, _ = get_ca_layers(final_pipe.unet, with_to_k=True)
    
    train_dataset = MACEDataset(
        tokenizer=lora_pipe.tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        use_pooler=args.use_pooler,
        multi_concept=args.multi_concept[0],
        mapping=args.mapping_concept,
        augment=args.augment,
        batch_size=args.train_batch_size,
        with_prior_preservation=args.with_prior_preservation,
        aug_length=args.aug_length,
        prompt_len=args.prompt_len,
        input_data_path=args.input_data_dir,
    )
    
    # to save memory
    CFR_dict = {}
    for layer_num in tqdm(range(len(final_projection_matrices))):
        CFR_dict[f'{layer_num}_for_mat1'] = None
        CFR_dict[f'{layer_num}_for_mat2'] = None
        
    all_contexts = []
    all_valuess = []
    all_concepts = []
    max_concept_num = args.max_memory # the maximum number of concept that can be processed at once
    count = 0
    for single_concept in train_dataset.dict_for_close_form:
        count += 1
        print(f"============================== Concept {count}: {single_concept['old'][0][1]} ==============================")
        all_concepts.append(single_concept['old'][0][1])
        lora_pipe.load_lora_weights(f"{model_id}/lora/{single_concept['old'][0][1].replace(' ', '-')}")
        lora_pipe.fuse_lora(lora_scale=1.0)
        
        lora_projection_matrices, lora_ca_layers, lora_og_matrices = get_ca_layers(lora_pipe.unet, with_to_k=True)

        contexts, valuess = prepare_k_v(lora_pipe.text_encoder, lora_projection_matrices, lora_ca_layers, lora_og_matrices, 
                                        [single_concept], lora_pipe.tokenizer, all_words=True, prepare_k_v_for_lora=True)

        # if the number of concept is too large, we need to use cache mode to save memory
        if len(train_dataset.dict_for_close_form) > max_concept_num:
            closed_form_refinement(lora_projection_matrices, contexts, valuess, cache_dict=CFR_dict, cache_mode=True)
        
            del contexts, valuess
            gc.collect()
            torch.cuda.empty_cache()
        else:
            all_contexts.append(contexts[0])
            all_valuess.append(valuess[0])
            
        lora_pipe.unfuse_lora()
        lora_pipe.unload_lora_weights()
    
    del lora_pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load cached prior knowledge for preserving
    if args.prior_preservation_cache_path:
        prior_preservation_cache_dict = torch.load(args.prior_preservation_cache_path, map_location=device)
    else:
        prior_preservation_cache_dict = {}
        for layer_num in tqdm(range(len(final_projection_matrices))):
            prior_preservation_cache_dict[f'{layer_num}_for_mat1'] = .0
            prior_preservation_cache_dict[f'{layer_num}_for_mat2'] = .0
            
    # Load cached domain knowledge for preserving
    if args.domain_preservation_cache_path:
        domain_preservation_cache_dict = torch.load(args.domain_preservation_cache_path, map_location=device)
    else:
        domain_preservation_cache_dict = {}
        for layer_num in tqdm(range(len(final_projection_matrices))):
            domain_preservation_cache_dict[f'{layer_num}_for_mat1'] = .0
            domain_preservation_cache_dict[f'{layer_num}_for_mat2'] = .0
    
    # integrate the preserving knowledge and multi-lora knowledge
    cache_dict = {}
    if len(train_dataset.dict_for_close_form) > max_concept_num:    
        for key in CFR_dict:
            cache_dict[key] = args.train_preserve_scale * (prior_preservation_cache_dict[key] \
                            + args.preserve_weight * domain_preservation_cache_dict[key]) \
                            + CFR_dict[key]
    
        closed_form_refinement(final_projection_matrices, lamb=args.lamb, preserve_scale=1, cache_dict=cache_dict)
    else:
        for key in prior_preservation_cache_dict:
            cache_dict[key] = prior_preservation_cache_dict[key] \
                            + args.preserve_weight * domain_preservation_cache_dict[key]
                            
        closed_form_refinement(final_projection_matrices, all_contexts, all_valuess, lamb=args.lamb, 
                               preserve_scale=args.fuse_preserve_scale, cache_dict=cache_dict)

    # save the final model
    final_pipe.save_pretrained(args.final_save_path)

    


