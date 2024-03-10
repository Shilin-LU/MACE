import os
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import pandas as pd
import argparse
from accelerate import PartialState, Accelerator


def generate_images(model_name, prompts_path, save_path, step, device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=100, num_samples=1, from_case=0):

    pipe = StableDiffusionPipeline.from_pretrained(model_name)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    df = pd.read_csv(prompts_path)
    
    accelerator = Accelerator()
    state = PartialState()
    pipe.to(state.device)

    for i in range(0, len(df), step): 
        
        if state.process_index == 0:
            idx = i
        elif state.process_index == 1:
            idx = i + 1
        elif state.process_index == 2:
            idx = i + 2
        elif state.process_index == 3:
            idx = i + 3
        elif state.process_index == 4:
            idx = i + 4
        elif state.process_index == 5:
            idx = i + 5
        elif state.process_index == 6:
            idx = i + 6
        elif state.process_index == 7:
            idx = i + 7
        
        if idx < len(df):
            row = df.iloc[idx]
        
        os.makedirs(f"{save_path}/{row.type}", exist_ok=True)
        
        prompt = [str(row.prompt)]*num_samples
        seed = row.evaluation_seed
        # print(f'Inferencing: {prompt}')

        # Check if the file exists in the given folder path
        folder_path = f"{save_path}/{row.type}"
        filename = f"{prompt[0]}_{row.evaluation_seed}.png"
        
        if os.path.isfile(os.path.join(folder_path, filename)):
            print(f"File {filename} exists.")
        else:
            print(f"File {filename} does not exist, running the function.")
            images = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=1, 
                          generator=torch.manual_seed(seed)).images
            for k, im in enumerate(images):
                im.save(f"{folder_path}/{filename}")   
    
        accelerator.wait_for_everyone()


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='name of model', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, 
                        required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, 
                        required=True)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:3')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--step', help='ddim steps of inference used to train', type=int, required=True)
    args = parser.parse_args()
    
    model_name = args.model_name
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples= args.num_samples
    from_case = args.from_case
    step = args.step
    
    generate_images(model_name, prompts_path, save_path, step, device=device, guidance_scale = guidance_scale, 
                    image_size=image_size, ddim_steps=ddim_steps, num_samples=num_samples,from_case=from_case)
