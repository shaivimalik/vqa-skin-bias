#!/bin/bash

# Base image directory
IMG_DIR="../GRAS_DS"

cd inference_scripts 

echo "Running paligemma2-3b-mix-224 inference for Gender, Race, and Age..."
python paligemma_fairface.py --img_dir $IMG_DIR --output_dir ../paligemma2-3b-mix-224_results

echo "Running paligemma2-3b-mix-224 inference for Skin Tone..."
python paligemma_aiface.py --img_dir $IMG_DIR --output_dir ../paligemma2-3b-mix-224_results

echo "Running Qwen2.5-VL-3B-Instruct inference for Gender, Race, and Age..."
python qwen_fairface.py --img_dir $IMG_DIR --output_dir ../Qwen2.5-VL-3B-Instruct_results

echo "Running Qwen2.5-VL-3B-Instruct inference for Skin Tone..."
python qwen_aiface.py --img_dir $IMG_DIR --output_dir ../Qwen2.5-VL-3B-Instruct_results

echo "Running blip2-opt-2.7b inference for Gender, Race, and Age..."
python blip_fairface.py --img_dir $IMG_DIR --output_dir ../blip2-opt-2.7b_results

echo "Running blip2-opt-2.7b inference for Skin Tone..."
python blip_aiface.py --img_dir $IMG_DIR --output_dir ../blip2-opt-2.7b_results

echo "Running llava-1.5-7b-hf inference for Gender, Race, and Age..."
python llava_fairface.py --img_dir $IMG_DIR --output_dir ../llava-1.5-7b-hf_results

echo "Running llava-1.5-7b-hf inference for Skin Tone..."
python llava_aiface.py --img_dir $IMG_DIR --output_dir ../llava-1.5-7b-hf_results

echo "Running Phi-4-multimodal-instruct inference for Gender, Race, and Age..."
python phi_fairface.py --img_dir $IMG_DIR --output_dir ../Phi-4-multimodal-instruct_results

echo "Running Phi-4-multimodal-instruct inference for Skin Tone..."
python phi_aiface.py --img_dir $IMG_DIR --output_dir ../Phi-4-multimodal-instruct_results

echo "All inference tasks completed!"