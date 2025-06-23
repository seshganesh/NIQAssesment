import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import LlavaForConditionalGeneration
from transformers import LlavaProcessor
# from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from PIL import Image
from pathlib import Path
import json

ROOT_DIR = Path.cwd()

config_json = {
    "model_name" :"llava-hf/llava-1.5-7b-hf",
    "output_dir" : "./llava-chef-model",
    "batch_size": 4,
    "learning_rate": 2e-5,
    "num_epochs" : 3,
    "warmup_steps": 500,
    "max_length" : 512,
    "image_size" : 224,
    "gradient_accumulation_steps": 4,
    "save_steps": 1000,
    "eval_steps" : 500,
    "logging_steps" : 100,
    "dataloader_num_workers" : 4 }


def sample_dataset(inp_json_path = ROOT_DIR / "data" / "ocr_data.json"):
    with open(inp_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_ingredients_data(data):
    result = []
    for id_key, items in data.items():
        for item in items:
            for key, value in item.items():
                if 'ingredient' in key.lower():
                    for file in os.listdir(ROOT_DIR / "data" / "images"):
                        if file.startswith(id_key) and "ingredient" in str(file):
                            image_name = os.path.join(ROOT_DIR / "data" / "images" / file)
                    it = {}
                    it["imagepath"] = image_name   # fix to match Dataset class
                    it["question"] = "what are the different ingredients"
                    it["answer"] = value.replace("\n", " ")
                    result.append(it)
    return result

class ingrDataset(Dataset):
    def __init__(self, data = None, processor = None , max_length = 512):
        super().__init__()
        self.processor = processor 
        self.max_length = max_length
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = item["imagepath"]
        image = Image.open(image_path).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": item['question']}
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": item['answer']}
                ]
            }
        ]

        text = self.processor.apply_chat_template(conversation, add_generation_prompt = False)

        inps = self.processor(
            text = text, 
            images = image, 
            return_tensors = "pt",
            max_length = self.max_length,
            truncation = True,
            padding = True
            
        )

        for key in inps: inps[key] = inps[key].squeeze(0)

        return inps
    

class llavatrainer:
    def __init__(self):
        self.config = config_json
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = LlavaProcessor.from_pretrained(self.config.get("model_name"))
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.config.get("model_name"),
            torch_dtype = torch.float16,
            device_map = "cuda"
        )
        self.model.train()

        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        
    def build_dataset(self, data):
        train_dataset = ingrDataset(
            data, 
            self.processor, 
            self.config.max_length
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["dataloader_num_workers"]
        )

        return train_dataloader

if __name__ == "__main__":
    json_dt = sample_dataset()
    json_ingred = get_ingredients_data(json_dt)

    llava_train = llavatrainer()

    train_loader = llava_train.build_dataset(json_ingred)
    print("dd")