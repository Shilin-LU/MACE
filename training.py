import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
from omegaconf import OmegaConf
import torch
from src.cfr_lora_training import main as cfr_lora_training
from src.fuse_lora_close_form import main as multi_lora_fusion
from inference import main as inference


def main(conf):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # stage 1 & 2 (CFR and LoRA training)
    cfr_lora_training(conf.MACE)

    # stage 3 (Multi-LoRA fusion)
    multi_lora_fusion(conf.MACE)

    # test the erased model
    if conf.MACE.test_erased_model:
        inference(OmegaConf.create({
            "pretrained_model_name_or_path": conf.MACE.final_save_path,
            "multi_concept": conf.MACE.multi_concept,
            "generate_training_data": False,
            "device": device,
            "steps": 50,
            "output_dir": conf.MACE.final_save_path,
        }))


if __name__ == "__main__":
    
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    main(conf)
