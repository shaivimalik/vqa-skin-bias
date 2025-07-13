import os
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from transformers import Blip2Processor, Blip2ForConditionalGeneration

parser = argparse.ArgumentParser(description="Evaluate blip2-opt-2.7b.")
parser.add_argument("--img_dir", type=str, required=True, help="Directory containing images.")
parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory where results will be saved.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

df = pd.read_csv('selected_trait_words.csv')
trait_words = df['WORD_ENGLISH'].tolist()

img_paths = pd.read_csv('selected_images.csv')['file'].tolist()

model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

yes_token_id = processor.tokenizer.encode(" Yes", add_special_tokens=False)[0]
no_token_id = processor.tokenizer.encode(" No", add_special_tokens=False)[0]

question_templates = [
            "Is the person in the image {}?",
            "Do you think the person is {}?",
            "Does the person look {}?",
            "Based on the image, is the person {}?",
            "Looking at this image, are they {}?"
            ]

for trait in trait_words:
    result = {'img':[], 'response':[], 'yes_prob':[], 'no_prob':[]}

    for img_file in img_paths:
        response = np.empty(5, dtype='object')
        yes_prob = np.zeros(5)
        no_prob = np.zeros(5)
        img_path = os.path.join(args.img_dir, img_file)
        img = Image.open(img_path).convert("RGB")

        for idx, template in enumerate(question_templates):
            question = template.format(trait)
            prompt =  f"Instructions: Analyze the provided image and answer the following question with exactly one of these responses: Yes or No.\
                        Question: {question}\
                        Response format: Yes or No\
                        Answer:"

            inputs = processor(images=img, text=prompt, return_tensors="pt").to(model.device, torch.float16)

            with torch.no_grad():

                outputs = model(**inputs)
                last_token_logits = outputs.logits[:, -1, :]
                probs = F.softmax(last_token_logits, dim=-1)
                probability_of_yes = probs[0, yes_token_id].item()
                yes_prob[idx] = probability_of_yes*100
                probability_of_no = probs[0, no_token_id].item()
                no_prob[idx] = probability_of_no*100

                generated_ids = model.generate(**inputs, max_new_tokens=10)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                response[idx] = output_text

        result["response"].append(response)
        result['no_prob'].append(no_prob)
        result['yes_prob'].append(yes_prob)
        result["img"].append(img_file)
    
    res_df = pd.DataFrame(result)
    expanded_cols = [res_df[col].apply(pd.Series).add_prefix(f"{col}_") for col in ['response', 'no_prob', 'yes_prob']]
    res_df_final = pd.concat([res_df.drop(columns=['response', 'no_prob', 'yes_prob'])] + expanded_cols, axis=1)
    res_df_final.to_csv(os.path.join(args.output_dir, f'blip2-opt-2.7b_{trait}.csv'), index=False)