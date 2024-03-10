import sys
import torch
from diffusers import StableDiffusionPipeline
from src.cfr_utils import *
import gc
import json


def extract_prompts(annotation_file):
    with open(annotation_file, 'r') as file:
        annotations = json.load(file)

    prompts = []
    for i, annotation in enumerate( annotations['annotations']):
        prompt = annotation['caption']
        prompts.append([prompt])

    return prompts


def main():   

    # file path
    train_annotation_file = './coco2014/annotations/captions_train2014.json'
    val_annotation_file = './coco2014/annotations/captions_val2014.json'

    # extract prompts
    train_prompts = extract_prompts(train_annotation_file)
    val_prompts = extract_prompts(val_annotation_file)
    total_prompts = train_prompts + val_prompts
    print(f"Number of prompts in training set: {len(train_prompts)}")
    print(f"Number of prompts in validation set: {len(val_prompts)}")
    
    model = "CompVis/stable-diffusion-v1-4"
    final_pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float32).to("cuda")
    final_projection_matrices, final_ca_layers, final_og_matrices = get_ca_layers(final_pipe.unet, with_to_k=True)
        
    cache_dict = {}
    for layer_num in tqdm(range(len(final_projection_matrices))):
        cache_dict[f'{layer_num}_for_mat1'] = None
        cache_dict[f'{layer_num}_for_mat2'] = None

    step = 500
    for i in range(0, len(total_prompts), step):
        entry = {"old": total_prompts[i:i+step], "new": total_prompts[i:i+step]}
    
        contexts, valuess = prepare_k_v(final_pipe.text_encoder, final_projection_matrices, final_ca_layers, 
                                        final_og_matrices, [entry], final_pipe.tokenizer, all_words=True)
        
        closed_form_refinement(final_projection_matrices, contexts, valuess, cache_dict=cache_dict, cache_mode=True)
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f'==================== num: {i}/{len(total_prompts)}====================')
        if i % 10000 == 0:
            torch.save(cache_dict, f"./cache/coco/cache_{i}.pt")
    
    torch.save(cache_dict, f"./cache/coco/cache_final.pt")
    

if __name__ == "__main__":
    main()

    

