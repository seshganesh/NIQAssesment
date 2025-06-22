import ollama
import base64
from pathlib import Path
import json
import re

ROOT_DIR = Path.cwd()
images_path = ROOT_DIR / "data" / "images"

with open(images_path / "8422947520014_front.32.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

def llava_description(imagefile):
    with open(imagefile, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    response = ollama.chat(
        model='llava',
        messages=[
            {
                "role": "user",
                "content": """What is this dish? Give me ONLY the ingredients for making the recipe. 
                Give me in the below format as a simple dictionary. Dont deviate from the dictionary format:
                {
                dish: 
                Ingredients :
                }
                """,
                "images": [image_base64]
            }
        ]
    )
    json_text = response['message']['content']
    # json_text = json_text.strip()
    # json_text = re.sub(r'^```json', '', json_text)
    # json_text = re.sub(r'^```python', '', json_text)
    # json_text = re.sub(r'^```', '', json_text)
    # json_text = re.sub(r'```$', '', json_text)
    # json_text = re.sub(r'^json\s*$', '', json_text)
    # json_text = re.sub(r'^python\s*$', '', json_text)
    # json_text = re.sub(r'(\s*)(\w+)\s*:', r'\1"\2":', json_text)
    # json_text = re.sub(r'"Ingredients"\s*:\s*\{', '"Ingredients": [', json_text)
    # json_text = re.sub(r'\}', ']', json_text, count=1)
    # json_text = re.sub(r'\]\s*$', '}', json_text)
    # json_text = json_text.strip()

    return json_text