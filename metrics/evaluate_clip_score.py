import os
from PIL import Image
import pandas as pd
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from argparse import ArgumentParser
import torch


@torch.no_grad()
def mean_clip_score(image_dir, prompts_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    text_df=pd.read_csv(prompts_path)
    texts=list(text_df['prompt'])
    image_filenames=os.listdir(image_dir)
    assert len(texts)==len(image_filenames), "Number of images and prompts don't match"
    
    sorted_image_filenames = sorted(image_filenames, key=lambda x: int(x.split("_")[0]))
    similarities=[]
    for i in tqdm(range(len(texts))):
        text=texts[i]
        imagename=sorted_image_filenames[i]
        image=Image.open(os.path.join(image_dir,imagename))
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
        outputs = model(**{k : v.to(device) for k, v in inputs.items()})
        clip_score= outputs.logits_per_image[0][0].detach().cpu()  # this is the image-text similarity score
        # print(text)
        # print(imagename)
        # print(clip_score)
        similarities.append(clip_score)
    similarities=np.array(similarities)
    
    mean_similarity=np.mean(similarities)
    std_similarity = np.std(similarities)

    print('-------------------------------------------------')
    print('\n')
    print(f"Mean CLIP score ± Standard Deviation: {mean_similarity:.4f}±{std_similarity:.4f}")   
    

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--image_dir", type=str, default='path/to/generated_images')
    parser.add_argument("--prompts_path", type=str, default='./prompts_csv/coco_30k.csv')
    args = parser.parse_args()

    image_dir=args.image_dir
    prompts_path=args.prompts_path
    
    mean_clip_score(image_dir, prompts_path)
