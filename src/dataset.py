from torch.utils.data import Dataset
from src.cfr_utils import *
from torchvision import transforms
from PIL import Image
import random
from pathlib import Path
import os
from openai import OpenAI
import regex as re


BASE_URL = ''
API_KEY = ''


def clean_prompt(class_prompt_collection):
    class_prompt_collection = [re.sub(
        r"[0-9]+", lambda num: '' * len(num.group(0)), prompt) for prompt in class_prompt_collection]
    class_prompt_collection = [re.sub(
        r"^\.+", lambda dots: '' * len(dots.group(0)), prompt) for prompt in class_prompt_collection]
    class_prompt_collection = [x.strip() for x in class_prompt_collection]
    class_prompt_collection = [x.replace('"', '') for x in class_prompt_collection]
    return class_prompt_collection


def text_augmentation(erased_concept, mapping_concept, concept_type, num_text_augmentations=100):
    
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    
    class_prompt_collection = []

    if concept_type == 'object':
        messages = [
            {"role": "system", "content": "You can describe any image via text and provide captions for wide variety of images that is possible to generate."},
            {"role": "user", "content": f"Generate {num_text_augmentations} captions for images containing {erased_concept}. The caption should also contain the word '{erased_concept}'. Please do not use any emojis in the captions."},
        ]
        
        while True:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )
            class_prompt_collection += [x for x in completion.choices[0].message.content.lower(
            ).split('\n') if erased_concept in x]
            messages.append(
                {"role": "assistant", "content": completion.choices[0].message.content})
            messages.append(
                {"role": "user", "content": f"Generate {num_text_augmentations-len(class_prompt_collection)} more captions"})
            if len(class_prompt_collection) >= num_text_augmentations:
                break
            
        class_prompt_collection = clean_prompt(class_prompt_collection)[:num_text_augmentations]
        class_prompt_formated = []
        mapping_prompt_formated = []
        
        for prompt in class_prompt_collection:
            class_prompt_formated.append((prompt, erased_concept))
            mapping_prompt_formated.append((prompt.replace(erased_concept, mapping_concept), mapping_concept))
    
        return class_prompt_formated, mapping_prompt_formated
        
        
class MACEDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        tokenizer,
        size=512,
        center_crop=False,
        use_pooler=False,
        multi_concept=None,
        mapping=None,
        augment=True,
        batch_size=None,
        with_prior_preservation=False,
        preserve_info=None,
        num_class_images=None,
        train_seperate=False,
        aug_length=50,
        prompt_len=250,
        input_data_path=None,
        use_gpt=False,
    ):  
        self.with_prior_preservation = with_prior_preservation
        self.use_pooler = use_pooler
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.batch_counter = 0
        self.batch_size = batch_size
        self.concept_number = 0
        self.train_seperate = train_seperate
        self.aug_length = aug_length
        
        self.all_concept_image_path  = []
        self.all_concept_mask_path  = []
        single_concept_images_path = []
        self.instance_prompt  = []
        self.target_prompt  = []
        
        self.num_instance_images = 0
        self.dict_for_close_form = []
        self.class_images_path = []
        
        for concept_idx, (data, mapping_concept) in enumerate(zip(multi_concept, mapping)):
            c, t = data
            
            if input_data_path is not None:
                p = Path(os.path.join(input_data_path, c.replace("-", " ")))
                if not p.exists():
                    raise ValueError(f"Instance {p} images root doesn't exists.")
                
                if t == "object":
                    p_mask = Path(os.path.join(input_data_path, c.replace("-", " ")).replace(f'{c.replace("-", " ")}', f'{c.replace("-", " ")} mask'))
                    if not p_mask.exists():
                        raise ValueError(f"Instance {p_mask} images root doesn't exists.")
            else:
                raise ValueError(f"Input data path is not provided.")    
            
            image_paths = list(p.iterdir())
            single_concept_images_path = []
            single_concept_images_path += image_paths
            self.all_concept_image_path.append(single_concept_images_path)
            
            if t == "object":
                mask_paths = list(p_mask.iterdir())
                single_concept_masks_path = []
                single_concept_masks_path += mask_paths
                self.all_concept_mask_path.append(single_concept_masks_path)
                     
            erased_concept = c.replace("-", " ")
            
            if use_gpt:
                class_prompt_collection, mapping_prompt_collection = text_augmentation(erased_concept, mapping_concept, t, num_text_augmentations=self.aug_length)
                self.instance_prompt.append(class_prompt_collection)
                self.target_prompt.append(mapping_prompt_collection)
            else: 
                sampled_indices = random.sample(range(0, prompt_len), self.aug_length)
                self.instance_prompt.append(prompt_augmentation(erased_concept, augment=augment, sampled_indices=sampled_indices, concept_type=t))
                self.target_prompt.append(prompt_augmentation(mapping_concept, augment=augment, sampled_indices=sampled_indices, concept_type=t))
                
            self.num_instance_images += len(single_concept_images_path)
            
            entry = {"old": self.instance_prompt[concept_idx], "new": self.target_prompt[concept_idx]}
            self.dict_for_close_form.append(entry)
            
        if with_prior_preservation:
            class_data_root = Path(preserve_info['preserve_data_dir'])
            if os.path.isdir(class_data_root):
                class_images_path = list(class_data_root.iterdir())
                class_prompt = [preserve_info["preserve_prompt"] for _ in range(len(class_images_path))]
            else:
                with open(class_data_root, "r") as f:
                    class_images_path = f.read().splitlines()
                with open(preserve_info["preserve_prompt"], "r") as f:
                    class_prompt = f.read().splitlines()

            class_img_path = [(x, y) for (x, y) in zip(class_images_path, class_prompt)]
            self.class_images_path.extend(class_img_path[:num_class_images])
                     
        self.image_transforms = transforms.Compose(
            [
                # transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        self._concept_num = len(self.instance_prompt)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_instance_images // self._concept_num, self.num_class_images)
        
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        
        if not self.train_seperate:
            if self.batch_counter % self.batch_size == 0:
                self.concept_number = random.randint(0, self._concept_num - 1)
            self.batch_counter += 1
        
        instance_image = Image.open(self.all_concept_image_path[self.concept_number][index % self._length])
        
        if len(self.all_concept_mask_path) == 0:
            # artistic style erasure
            binary_tensor = None
        else:
            # object/celebrity erasure
            instance_mask = Image.open(self.all_concept_mask_path[self.concept_number][index % self._length])
            instance_mask = instance_mask.convert('L')
            trans = transforms.ToTensor()
            binary_tensor = trans(instance_mask)
        
        prompt_number = random.randint(0, len(self.instance_prompt[self.concept_number]) - 1)
        instance_prompt, target_tokens = self.instance_prompt[self.concept_number][prompt_number]
        
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_prompt"] = instance_prompt
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_masks"] = binary_tensor

        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        prompt_ids = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length
        ).input_ids

        concept_ids = self.tokenizer(
            target_tokens,
            add_special_tokens=False
        ).input_ids             

        pooler_token_id = self.tokenizer(
            "<|endoftext|>",
            add_special_tokens=False
        ).input_ids[0]

        concept_positions = [0] * self.tokenizer.model_max_length
        for i, tok_id in enumerate(prompt_ids):
            if tok_id == concept_ids[0] and prompt_ids[i:i + len(concept_ids)] == concept_ids:
                concept_positions[i:i + len(concept_ids)] = [1]*len(concept_ids)
            if self.use_pooler and tok_id == pooler_token_id:
                concept_positions[i] = 1
        example["concept_positions"] = torch.tensor(concept_positions)[None]               

        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["preserve_images"] = self.image_transforms(class_image)
            example["preserve_prompt_ids"] = self.tokenizer(
                class_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
            
        return example