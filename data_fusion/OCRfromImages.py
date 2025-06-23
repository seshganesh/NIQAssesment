import cv2
import pytesseract
from PIL import Image
from pathlib import Path
import os
import re
import json
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from MLLLMIngredientsfromImages import llava_description
import random
os.environ['PYDEVD_EVALUATION_TIMEOUT_SEC'] = '120'
ROOT_DIR = Path.cwd()

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_ocr(image_path):
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(image)
        # try:
        #     ingre_dict = llava_description(image_path)
        # except:
        #     return text, ""
        return text
    except: 
        return ""
    
def model_translator():
    model_name = 'Helsinki-NLP/opus-mt-mul-en'  # mul-en = many languages to English

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # text = "Bonjour, comment ça va ?"
    text = "SIMPly Steam Or pain 4ySainsbury’s Sept KOO87B R me by JMoDonalda OO ape"
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    translated = model.generate(**inputs)
    output = tokenizer.decode(translated[0], skip_special_tokens=True)

    print(output)

    
def get_ocr_all_images(images_path, filtered_images):
    filt_image_dict = {}
    images_len = 0
    for f in filtered_images:
        code = f.split("_")[0]
        image_name = f.split("_")[1]

        ocr_text= get_ocr(images_path / f)
        if code not in filt_image_dict.keys():
            filt_image_dict.update({code:[{image_name: ocr_text}]})
        else:
            filt_image_dict[code].append({image_name: ocr_text})
        images_len += len(filt_image_dict.keys())
        if images_len > 30:
            return filt_image_dict


def extract_ocr_from_images(images_path):
    matching_files = [f for f in os.listdir(images_path)]
    filtered_images = [f for f in matching_files if not re.search(r'_\d+\.0\.jpg$', f)] # remove invalid images
    filt_image_dict = get_ocr_all_images(images_path, filtered_images)
    with open("data/ocr_data.json", "w", encoding="utf-8") as f:
        json.dump(filt_image_dict, f, ensure_ascii=False, indent = 4)
    return filt_image_dict

def convertjsontodataframe(inp_dict):
    records = []
    for code, entries in data.items():
        record = {"code": code, "front": None, "ingredients": None, "nutriments": None}
        for item in entries:
            for key, value in item.items():
                key_lower = key.lower()
                if "front" in key_lower and record["front"] is None:
                    record["front"] = value.strip()
                elif "ingredient" in key_lower and record["ingredients"] is None:
                    record["ingredients"] = value.strip()
                elif  "nutrition" in key_lower and record["nutriments"] is None: 
                    record["nutriments"] = value.strip()
        records.append(record)
    return pd.DataFrame(records)
    

def split_text(txt):
    if pd.isna(txt):
        return []
    toks = re.findall(r'\n|[^\s\n]+', txt)
    return toks

def convertoBIO(data):
    df_data=convertjsontodataframe(data)
    df_data.to_csv(ROOT_DIR / "data" / "df_bio.csv")
    bio_rows = []
    for idx, row in  df_data.iterrows():
        for col in ["front", "ingredients",  "nutriments"]:
            tokens = split_text(row[col])
            start = True
            tag_list = ["ITEM", "INGREDIENT", "INGRED_VALUE", "NUTRIMENT", "NUTRIMENT_VALUE"]
            tag_random = random.choice(tag_list)
            for token in tokens:
                token_clean = token.strip()
                if token_clean == "":
                    continue
                if token_clean == "\n":
                    tag = "O"
                else: 
                    tag = f"B-{tag_random}" if start else f"I-{tag_random}"
                    start = False
                bio_rows.append([str(token_clean)+" "+str(col)+" "+str(tag)+"\n"])
        bio_rows.append(["\t\n"])

    with open(ROOT_DIR / "data" / "bio_output.txt", "w", encoding="utf-8") as f:    
        for entry in bio_rows:
            f.write(entry[0])
  


if __name__ == "__main__":
    images_path = ROOT_DIR / "data" / "images"
    ocr_extracted_images = extract_ocr_from_images(images_path) 
    with open(ROOT_DIR / "data"/ "ocr_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    convertoBIO(data)
    


