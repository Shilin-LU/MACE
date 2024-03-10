import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from omegaconf import OmegaConf
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from inference import main as inference
from src.grounded_sam_util import get_mask, load_model
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)


def main(conf):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # generate 8 images per concept using the original model for performing erasure
    if conf.MACE.generate_data:
        inference(OmegaConf.create({
            "pretrained_model_name_or_path": 'CompVis/stable-diffusion-v1-4',
            "multi_concept": conf.MACE.multi_concept,
            "generate_training_data": True,
            "device": device,
            "steps": 30,
            "output_dir": conf.MACE.input_data_dir,
        }))

    # get and save masks for each image
    if conf.MACE.use_gsam_mask:
        grounded_model = load_model(conf.MACE.grounded_config, conf.MACE.grounded_checkpoint, device=device)
        
        if conf.MACE.use_sam_hq:
            predictor = SamPredictor(sam_hq_model_registry['vit_h'](checkpoint=conf.MACE.sam_hq_checkpoint).to(device))
        else:
            predictor = SamPredictor(sam_model_registry['vit_h'](checkpoint=conf.MACE.sam_checkpoint).to(device))
        
        transform = transforms.ToTensor()
        for root, _, files in os.walk(conf.MACE.input_data_dir):
            mask_save_path = root.replace(f'{os.path.basename(root)}', f'{os.path.basename(root)} mask')
            os.makedirs(mask_save_path, exist_ok=True)
            for file in files:
                file_path = os.path.join(root, file)
                print(file_path)
                # read images and get masks
                image = Image.open(file_path)
                if not image.mode == "RGB":
                    image = image.convert("RGB")
                tensor_image = transform(image).to(device)
                GSAM_mask = get_mask(tensor_image, os.path.basename(root), grounded_model, predictor, device)
                # save masks
                GSAM_mask = (GSAM_mask.to(torch.uint8) * 255).squeeze()
                save_mask = to_pil_image(GSAM_mask)
                save_mask.save(f"{os.path.join(mask_save_path, file).replace('.jpg', '_mask.jpg')}")
                

if __name__ == "__main__":
    
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    main(conf)