# JarvisArt Data Scripts

Image editing training data generation pipeline.

## Dependencies

```bash
pip install transformers accelerate qwen-vl-utils vllm tqdm openai google-generativeai pillow
```

## Modules

```
data_scripts/
├── image_pairs_xmp_generation/    # Generate image pairs and XMP configs
├── instructions_generation/       # Generate user instructions  
├── cot_generation/                # Generate expert reasoning
└── format_conversion/             # Convert to training format
```

## Usage

### 1. Generate Image Pairs and XMP Configs

```bash
cd image_pairs_xmp_generation/
bash run_recommendation.sh
```

### 2. Generate User Instructions

```bash
cd instructions_generation/
python instructions_gen.py \
    --base_path /path/to/data \
    --start_id 0 \
    --end_id 100 \
    --key YOUR_GEMINI_API_KEY
```

Outputs: `user_want.txt`, `user_want_2.txt`, `user_want_4.txt`, `user_want_7.txt`

### 3. Generate Expert Chain-of-Thought

```bash
cd cot_generation/
python cot_initial_gen.py \
    --base_path /path/to/data \
    --start_id 0 \
    --end_id 100 \
    --key YOUR_GOOGLE_API_KEY

python cot_refined_gen.py \
    --base_path /path/to/data \
    --start_id 0 \
    --end_id 100 \
    --key YOUR_GOOGLE_API_KEY
```

Outputs: `original_expert_cot.txt`, `revised160_expert_cot.txt`

### 4. Convert to Training Format

```bash
cd format_conversion/
python format_converter.py \
    --workers 100 \
    --tasks global_cot local_cot \
    --global_data_path /path/to/global/data \
    --local_data_path /path/to/local/data \
    --output_dir ./output \
    --before_image before.jpg \
    --roc_file config.lua \
    --user_intent_file user_want.txt \
    --cot_file revised160_expert_cot.txt
```

## Data Structure

Each image folder contains:
```
image_folder/
├── before.jpg                    # Original image
├── processed.jpg                 # Edited image
├── config.lua                    # Processing config
├── config.xmp                    # XMP metadata
├── user_want*.txt               # User instructions (4 variants)
├── original_expert_cot.txt      # Initial expert reasoning
└── revised160_expert_cot.txt    # Refined expert reasoning
```

## Parameters

- `--base_path`: Path to image folders
- `--start_id`: Start processing from this ID (0-based)
- `--end_id`: Stop at this ID (-1 for all remaining)
- `--key`: API key for Gemini/Google services
- `--workers`: Parallel threads for format conversion

## Complete Workflow

```bash
# 1. Generate configs
cd image_pairs_xmp_generation/ && bash run_recommendation.sh

# 2. Generate instructions
cd ../instructions_generation/
python vllm_api_instructions_gen.py --base_path ./data --start_id 0 --end_id 1000 --key YOUR_API_KEY

# 3. Generate reasoning
cd ../cot_generation/
python cot_initial_gen.py --base_path ./data --start_id 0 --end_id 1000 --key YOUR_API_KEY
python cot_refined_gen.py --base_path ./data --start_id 0 --end_id 1000 --key YOUR_API_KEY

# 4. Convert format
cd ../format_conversion/
python format_converter.py --workers 50 --tasks global_cot local_cot \
    --global_data_path ./data --output_dir ./output
```