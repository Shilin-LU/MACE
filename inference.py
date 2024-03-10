import os, gc
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from omegaconf import OmegaConf
import argparse


def main(args):

    model_id = args.pretrained_model_name_or_path
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(args.device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    torch.Generator(device=args.device).manual_seed(42)
    
    if args.generate_training_data:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        num_images = 8
        count = 0
        for single_concept in args.multi_concept:
            for c, t in single_concept:
                count += 1
                print(f"Generating training data for concept {count}: {c}...")
                c = c.replace('-', ' ')
                output_folder = f"{args.output_dir}/{c}"
                os.makedirs(output_folder, exist_ok=True)
                if t == "object":
                    prompt = f"a photo of the {c}"
                    print(f'Inferencing: {prompt}')
                    images = pipe(prompt, num_inference_steps=args.steps, guidance_scale=7.5, num_images_per_prompt=num_images).images
                    for i, im in enumerate(images):
                        im.save(f"{output_folder}/{prompt.replace(' ', '-')}_{i}.jpg")
                elif t == "style":
                    prompt = f"a photo in the style of {c}"
                    print(f'Inferencing: {prompt}')
                    images = pipe(prompt, num_inference_steps=args.steps, guidance_scale=7.5, num_images_per_prompt=num_images).images
                    for i, im in enumerate(images):
                        im.save(f"{output_folder}/{prompt.replace(' ', '-')}_{i}.jpg")
                else:
                    raise ValueError("unknown concept type.")
                del images
                torch.cuda.empty_cache()
                gc.collect()
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        num_images = args.num_images
        output_folder = f"{args.output_dir}/generated_images"
        os.makedirs(output_folder, exist_ok=True)
        print(f"Inference using {args.pretrained_model_name_or_path}...")
        prompt = args.prompt
        images = pipe(prompt, num_inference_steps=args.steps, guidance_scale=7.5, num_images_per_prompt=num_images).images
        for i, im in enumerate(images):
            im.save(f"{output_folder}/o_{prompt.replace(' ', '-')}_{i}.jpg")  
        
        torch.cuda.empty_cache()
        gc.collect()

    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_images', type=int, default=3)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    steps = 30
    model_id = args.model_path
    output_dir = args.save_path
    num_images = args.num_images
    prompt = args.prompt
    
    main(OmegaConf.create({
        "pretrained_model_name_or_path": model_id,
        "generate_training_data": False,
        "device": device,
        "steps": steps,
        "output_dir": output_dir,
        "num_images": num_images,
        "prompt": prompt,
    }))