from argparse import Namespace
import logging
import math
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from src.mace_lora_atten_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from src.cfr_utils import *
from src.dataset import MACEDataset
import json


logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    concept_positions = [example["concept_positions"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    masks = [example["instance_masks"] for example in examples]
    instance_prompts =  [example["instance_prompt"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["preserve_prompt_ids"] for example in examples]
        pixel_values += [example["preserve_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    if masks[0] is not None: 
        ## object/celebrity erasure
        masks = torch.stack(masks)
    else:
        ## artistic style erasure
        masks = None
    
    input_ids = torch.cat(input_ids, dim=0)
    concept_positions = torch.cat(concept_positions, dim=0).type(torch.BoolTensor)

    batch = {
        "instance_prompts": instance_prompts,
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "masks": masks,
        "concept_positions": concept_positions,
    }
    return batch


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if args.train_text_encoder and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )

    # projection_matrices, ca_layers, og_matrices = get_ca_layers(unet, with_to_k=True)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    if args.with_prior_preservation:
        args.preservation_info = {
                "preserve_prompt": args.preserve_prompt,
                "preserve_data_dir": args.preserve_data_dir
            }
    else:
        args.preservation_info = None
    
    train_dataset = MACEDataset(
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        use_pooler=args.use_pooler,
        multi_concept=args.multi_concept[0],
        mapping=args.mapping_concept,
        augment=args.augment,
        batch_size=args.train_batch_size,
        with_prior_preservation=args.with_prior_preservation,
        preserve_info=args.preservation_info,
        train_seperate=args.train_seperate,
        aug_length=args.aug_length,
        prompt_len=args.prompt_len,
        input_data_path=args.input_data_dir,
        use_gpt=args.use_gpt,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        
    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # stage 1: closed-form refinement
    projection_matrices, ca_layers, og_matrices = get_ca_layers(unet, with_to_k=True)
    
    # to save memory
    CFR_dict = {}
    max_concept_num = args.max_memory # the maximum number of concept that can be processed at once
    if len(train_dataset.dict_for_close_form) > max_concept_num:
        
        for layer_num in tqdm(range(len(projection_matrices))):
            CFR_dict[f'{layer_num}_for_mat1'] = None
            CFR_dict[f'{layer_num}_for_mat2'] = None
            
        for i in tqdm(range(0, len(train_dataset.dict_for_close_form), max_concept_num)):
            contexts_sub, valuess_sub = prepare_k_v(text_encoder, projection_matrices, ca_layers, og_matrices, 
                                                    train_dataset.dict_for_close_form[i:i+5], tokenizer, all_words=args.all_words)
            closed_form_refinement(projection_matrices, contexts_sub, valuess_sub, cache_dict=CFR_dict, cache_mode=True)
            
            del contexts_sub, valuess_sub
            gc.collect()
            torch.cuda.empty_cache()
            
    else:
        for layer_num in tqdm(range(len(projection_matrices))):
            CFR_dict[f'{layer_num}_for_mat1'] = .0
            CFR_dict[f'{layer_num}_for_mat2'] = .0
            
        contexts, valuess = prepare_k_v(text_encoder, projection_matrices, ca_layers, og_matrices, 
                                        train_dataset.dict_for_close_form, tokenizer, all_words=args.all_words)
    
    del ca_layers, og_matrices

    # Load cached prior knowledge for preserving
    if args.prior_preservation_cache_path:
        prior_preservation_cache_dict = torch.load(args.prior_preservation_cache_path, map_location=projection_matrices[0].weight.device)
    else:
        prior_preservation_cache_dict = {}
        for layer_num in tqdm(range(len(projection_matrices))):
            prior_preservation_cache_dict[f'{layer_num}_for_mat1'] = .0
            prior_preservation_cache_dict[f'{layer_num}_for_mat2'] = .0
            
    # Load cached domain knowledge for preserving
    if args.domain_preservation_cache_path:
        domain_preservation_cache_dict = torch.load(args.domain_preservation_cache_path, map_location=projection_matrices[0].weight.device)
    else:
        domain_preservation_cache_dict = {}
        for layer_num in tqdm(range(len(projection_matrices))):
            domain_preservation_cache_dict[f'{layer_num}_for_mat1'] = .0
            domain_preservation_cache_dict[f'{layer_num}_for_mat2'] = .0
    
    # integrate the prior knowledge, domain knowledge and closed-form refinement
    cache_dict = {}
    for key in CFR_dict:
        cache_dict[key] = args.train_preserve_scale * (prior_preservation_cache_dict[key] \
                        + args.preserve_weight * domain_preservation_cache_dict[key]) \
                        + CFR_dict[key]
    
    # closed-form refinement
    projection_matrices, _, _ = get_ca_layers(unet, with_to_k=True)
    
    if len(train_dataset.dict_for_close_form) > max_concept_num:
        closed_form_refinement(projection_matrices, lamb=args.lamb, preserve_scale=1, cache_dict=cache_dict)
    else:
        closed_form_refinement(projection_matrices, contexts, valuess, lamb=args.lamb, 
                               preserve_scale=args.train_preserve_scale, cache_dict=cache_dict)
    
    del contexts, valuess, cache_dict
    gc.collect()
    torch.cuda.empty_cache()
    
    # stage 2: multi-lora training
    for i in range(train_dataset._concept_num): # the number of concept/lora
        
        attn_controller = AttnController()
        if i != 0:
            unet.set_default_attn_processor()
        for name, m in unet.named_modules():
            if name.endswith('attn2') or name.endswith('attn1'):
                cross_attention_dim = None if name.endswith("attn1") else unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]

                m.set_processor(LoRAAttnProcessor(
                    hidden_size=hidden_size, 
                    cross_attention_dim=cross_attention_dim, 
                    rank=args.rank, 
                    attn_controller=attn_controller, 
                    module_name=name, 
                    preserve_prior=args.with_prior_preservation,
                ))

        ### set lora
        # unet.set_attn_processor(lora_attn_procs)
        lora_attn_procs = {}
        for key, value in zip(unet.attn_processors.keys(), unet.attn_processors.values()):
            if key.endswith("attn2.processor"):
                lora_attn_procs[f'{key}.to_k_lora'] = value.to_k_lora
                lora_attn_procs[f'{key}.to_v_lora'] = value.to_v_lora
                # lora_attn_procs[f'{key}.to_q_lora'] = value.to_q_lora
                # lora_attn_procs[f'{key}.to_out_lora'] = value.to_out_lora
        
        lora_layers = AttnProcsLayers(lora_attn_procs)

        optimizer = optimizer_class(
            lora_layers.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )
        
        if args.train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )
        
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("MACE")

        # Train
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(args.output_dir, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * args.gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

        if args.importance_sampling:
            print("""Using relation-focal importance sampling, which can make training more efficient
                  and is particularly beneficial in erasing mass concepts with overlapping terms.""")
            
            list_of_candidates = [
                x for x in range(noise_scheduler.config.num_train_timesteps)
            ]
            prob_dist = [
                importance_sampling_fn(x)
                for x in list_of_candidates
            ]
            prob_sum = 0
            # normalize the prob_list so that sum of prob is 1
            for j in prob_dist:
                prob_sum += j
            prob_dist = [x / prob_sum for x in prob_dist]
        
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
    
        debug_once = True
                
        if args.train_seperate:
            train_dataset.concept_number = i 
        for epoch in range(first_epoch, args.num_train_epochs):
            unet.train()
            if args.train_text_encoder:
                text_encoder.train()
                
            torch.cuda.empty_cache()
            gc.collect()
            
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step           
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with accelerator.accumulate(unet):
                    # show
                    if debug_once:
                        print('==================================================================')
                        print(f'Concept {i}: {batch["instance_prompts"][0]}')
                        print('==================================================================')
                        debug_once = False
                        
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    
                    if args.importance_sampling:
                        timesteps = np.random.choice(
                            list_of_candidates,
                            size=bsz,
                            replace=True,
                            p=prob_dist)
                        timesteps = torch.tensor(timesteps).cuda()
                    else:
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                        
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    if args.no_real_image:
                        noisy_latents = noise_scheduler.add_noise(torch.zeros_like(noise), noise, timesteps)                
                    else:
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    
                    # set concept_positions for this batch 
                    if args.use_gsam_mask:
                        GSAM_mask = batch['masks']
                    else:
                        GSAM_mask = None
                    
                    attn_controller.set_concept_positions(batch["concept_positions"], GSAM_mask, use_gsam_mask=args.use_gsam_mask)

                    # Predict the noise residual
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    
                    loss = attn_controller.loss()
                    
                    if args.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                        
                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                        
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        # params_to_clip = params_to_optimize
                        params_to_clip = lora_layers.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                    attn_controller.zero_attn_probs()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

        # Create the pipeline using using the trained modules and save it.
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            ## save lora layers
            if args.train_seperate:
                concepts, _ = args.multi_concept[0][i]
            else:
                concepts = len(args.multi_concept[0])
                
            unet = accelerator.unwrap_model(unet).to(torch.float32)
            lora_path = f"{args.output_dir}/lora/{concepts}"
            os.makedirs(lora_path, exist_ok=True)
            unet.save_attn_procs(lora_path)
            
            if isinstance(args, Namespace):
                with open(f"{args.output_dir}/my_args.json", "w") as f:
                    json.dump(vars(args), f, indent=4)    

        accelerator.end_training()
        
        del lora_attn_procs, lora_layers, optimizer, lr_scheduler, attn_controller
        torch.cuda.empty_cache()

        if not args.train_seperate:
            break
    
    # save base initialized model 
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=accelerator.unwrap_model(unet),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        revision=args.revision,
    )
    pipeline.save_pretrained(args.output_dir)
    