import os
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Evaluate llava-1.5-7b-hf.")
parser.add_argument("--img_dir", type=str, required=True, help="Directory containing images.")
parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory where results will be saved.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

df = pd.read_csv('../selected_trait_words.csv')
trait_words = df['WORD_ENGLISH'].tolist()

img_paths = pd.read_csv('../selected_images_aiface.csv')['file'].tolist()

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", use_fast=True)

yes_token_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]

question_templates = [
            "Is the person in the image {}?",
            "Do you think the person is {}?",
            "Does the person look {}?",
            "Based on the image, is the person {}?",
            "Can this person be described as {}?"
            ]

for trait in tqdm(trait_words, desc='processing traits'):
    result = {'img':[], 'yes_prob':[]}

    for img_file in tqdm(img_paths, desc=f'processing image for {trait}', leave=False):
        yes_prob = np.zeros(5)
        img_path = os.path.join(args.img_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((256,256))

        for idx, template in enumerate(question_templates):
            question = template.format(trait)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": f"Instructions: Analyze the provided image and answer the following question with exactly one of these responses: Yes or No.\
                                                  Question: {question}\
                                                  Response format: Yes or No"},
                    ],
                },
            ]
            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, torch.float16)

            with torch.no_grad():
                outputs = model(**inputs)
                last_token_logits = outputs.logits[:, -1, :]
                probs = F.softmax(last_token_logits, dim=-1)
                probability_of_yes = probs[0, yes_token_id].item()
                yes_prob[idx] = probability_of_yes
                
        result['yes_prob'].append(yes_prob)
        result["img"].append(img_file)
    
    res_df = pd.DataFrame(result)
    expanded_cols = [res_df[col].apply(pd.Series).add_prefix(f"{col}_") for col in ['yes_prob']]
    res_df_final = pd.concat([res_df.drop(columns=['yes_prob'])] + expanded_cols, axis=1)
    res_df_final.to_csv(os.path.join(args.output_dir, f'llava-1.5-7b-hf_{trait}_aiface.csv'), index=False)