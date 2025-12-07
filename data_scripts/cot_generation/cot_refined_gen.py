import os
from openai import OpenAI
import base64
import json
import tqdm
import argparse
from google import genai
from google.genai import types

# Prompt template for refine CoT with local coordinate marking
prompt_base_coor = """
Please revise the provided Chain of Thought(CoT) to follow these guidelines:

1. Thinking basis: Use user goals and original images instead of config files to generate all reasoning steps. For example: Avoid using words that are obviously related to configuration, such as "config files".;
2. Expressive style: Imitates the human-like language patterns used to express thought processes during interpersonal communication. For example, it needs to start with "Alright", and the sentences need to be connected as naturally as human speech.;
3. Enforce length constraints: Compress CoT narratives to under 160 words through strategic distillation of key reasoning components;
4. User intent analysis: The main content of the user intent analysis in the original CoT needs to be retained;
5. Original Image Analysis: The primary elements of the original image analysis from the initial thought process should be preserved, encompassing both content analysis and aesthetic evaluation.

Apply these rules rigorously to ensure that the final CoT accurately reflects the professional thought process of image processing experts when they only have the original image materials and user needs.

Special Notes: 
1. Output the modified CoT directly without the introduction of words such as "Alright, here's a refined Chain of Thought (CoT) that strictly adheres to the guidelines"
2. All outputs must be strictly in English, prohibiting the use of any other languages.
3. All image processing operations involved should NOT retain the specific adjustment values mentioned in the original chain of thought (CoT).
4. When mentioning coordinates or bounding boxes, you must enclose them within <box></box> tags.  For example: 'I might consider adding a subtle glow or halo around the person <box>0.1242, 0.699, 0.7453, 1.0000</box>'
5. <box></box> tags must be in the form of <box>x1 y1 x2 y2</box> to represent bounding boxes. It is important to note that the form of <box>x1 y1</box> is incorrect.
6. Please strictly enforce the word limit.
"""

# Prompt template for refine CoT
prompt_base_no = """
Please revise the provided Chain of Thought(CoT) to follow these guidelines:

1. Thinking basis: Use user goals and original images instead of config files to generate all reasoning steps. For example: Avoid using words that are obviously related to configuration, such as "config files".;
2. Expressive style: Imitates the human-like language patterns used to express thought processes during interpersonal communication. For example, it needs to start with "Alright", and the sentences need to be connected as naturally as human speech.;
3. Enforce length constraints: Compress CoT narratives to under 160 words through strategic distillation of key reasoning components;
4. User intent analysis: The main content of the user intent analysis in the original CoT needs to be retained;
5. Original Image Analysis: The primary elements of the original image analysis from the initial thought process should be preserved, encompassing both content analysis and aesthetic evaluation.

Apply these rules rigorously to ensure that the final CoT accurately reflects the professional thought process of image processing experts when they only have the original image materials and user needs.

Special Notes: 
1. Output the modified CoT directly without the introduction of words such as "Alright, here's a refined Chain of Thought (CoT) that strictly adheres to the guidelines"
2. All outputs must be strictly in English, prohibiting the use of any other languages.
3. All image processing operations involved should NOT retain the specific adjustment values mentioned in the original chain of thought (CoT).
4. Please strictly enforce the word limit.
"""

# Command line argument parser
parser = argparse.ArgumentParser(description="CoT revision script")

# Add arguments
parser.add_argument("--base_path", type=str, default="dataset/sample_data", help="Root directory of the dataset")
parser.add_argument("--start_id", type=int, default=0, help="Starting ID for processing")
parser.add_argument("--end_id", type=int, default=-1, help="Ending ID for processing")
parser.add_argument("--key", type=str, required=True, help="Google API key")

# Parse arguments
args = parser.parse_args()

base_path = args.base_path
start_id = args.start_id
end_id = args.end_id
key = args.key

def read_txt(path):
    """Read text file content"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content


# Initialize Google Gemini client
client = genai.Client(api_key=key)

# Get and sort folder list
path_name_list = os.listdir(base_path)
path_name_list.sort()

# Filter folders based on start and end IDs
path_name_list = path_name_list[start_id : end_id] if end_id != -1 else path_name_list[start_id : ]

# Process each folder
for path_name in tqdm.tqdm(path_name_list):
    try:
        # Build path
        preset_path_base = os.path.join(base_path, path_name)
        ori_expert_cot_path = os.path.join(preset_path_base, "original_expert_cot.txt")
        
        # Skip if revised CoT file already exists
        if os.path.exists(os.path.join(preset_path_base, "revised160_expert_cot.txt")):
            continue
            
        # Read original CoT content
        cot = read_txt(ori_expert_cot_path)
        
        # Choose prompt template based on whether coordinate box information exists
        if os.path.exists(os.path.join(preset_path_base, "box.txt")) and "portrait" in base_path:
            prompt_base = prompt_base_coor
            text_local = read_txt(os.path.join(preset_path_base, "box.txt"))
            prompt_user = "CoT: " + cot + "\nRelative coordinates of the region of interest:" + "<box>" + text_local + "</box>"
        else:
            prompt_base = prompt_base_no
            prompt_user = "CoT: " + cot
            
        # Build complete prompt
        messages = "a senior image processing specialist." + prompt_base + prompt_user
        
        # Call Gemini API to generate revised content
        response = client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=messages,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT']
                )
            )
            
        # Save revised CoT
        if response.text:
            with open(os.path.join(preset_path_base, "revised160_expert_cot.txt"), "w", encoding="utf-8") as fwrite:
                fwrite.write(response.text)
    except KeyboardInterrupt:
        print("Processing interrupted by user.")
        break
    except:
        print("Error occurred while generating 'cot revise' data. Error data:", path_name)