import os
from openai import OpenAI
import base64
import json
import tqdm
import argparse
from PIL import Image
from google import genai
from google.genai import types

# Prompt template for inital CoT with local coordinate marking
prompt_base_coor = """please analyze the provided before/after images, user requirements, Relative coordinates of the region of interest(optional), and configuration file to generate a detailed adjustment workflow. Although the configuration file is recognized, the response should avoid explicitly stating that the adjusted parameters were derived from it. The tone should convey a sense of expert judgment and reasoned analysis.
1. Technical Breakdown : Specify the tool or method to be utilized (e.g., global adjustment, tone curve adjustment, HSL adjustment, masking, texture, grain, cropping, dot) along with the specific value or range of values for the adjustment associated with that tool or method (e.g., "Saturation +15%", "High-pass Filter Radius: 3px"). All details regarding the tools/methods and their corresponding tuning values must be derived exclusively from the configuration file.
2. Step-by-Step Explanation : Describe adjustments in a logical sequence, prioritizing critical modifications first. Use layman-friendly terms but include professional jargon where necessary (e.g., 'recovered highlights via luminance mask').
3. Rationale : explain how each change aligns with the user's intent (e.g., 'cooling tone applied to match requested 'cinematic mood'').
Output Format : Freeform paragraphs with bullet points or numbered steps—no markdown. Prioritize clarity and technical precision. 

Special Notes: 
1. Respond EXCLUSIVELY in English for all outputs. Never include non-English characters or translations in other languages.
2. When mentioning coordinates or bounding boxes, you must enclose them within <box></box> tags. For example: 'I might consider adding a subtle glow or halo around the person (<box>0.1242, 0.699, 0.7453, 1.0000</box>)'
3. <box></box> tags must be in the form of <box>x1 y1 x2 y2</box> to represent bounding boxes. It is important to note that the form of <box>x1 y1</box> is incorrect.

"""

# Prompt template for inital CoT 
prompt_base_no = """please analyze the provided before/after images, user requirements, Relative coordinates of the region of interest(optional), and configuration file to generate a detailed adjustment workflow. Although the configuration file is recognized, the response should avoid explicitly stating that the adjusted parameters were derived from it. The tone should convey a sense of expert judgment and reasoned analysis.
1. Technical Breakdown : Specify the tool or method to be utilized (e.g., global adjustment, tone curve adjustment, HSL adjustment, masking, texture, grain, cropping, dot) along with the specific value or range of values for the adjustment associated with that tool or method (e.g., "Saturation +15%", "High-pass Filter Radius: 3px"). All details regarding the tools/methods and their corresponding tuning values must be derived exclusively from the configuration file.
2. Step-by-Step Explanation : Describe adjustments in a logical sequence, prioritizing critical modifications first. Use layman-friendly terms but include professional jargon where necessary (e.g., 'recovered highlights via luminance mask').
3. Rationale : explain how each change aligns with the user's intent (e.g., 'cooling tone applied to match requested 'cinematic mood'').
Output Format : Freeform paragraphs with bullet points or numbered steps—no markdown. Prioritize clarity and technical precision. 

Special Notes: 
    - Respond EXCLUSIVELY in English for all outputs. Never include non-English characters or translations in other languages.
"""

# Command line argument parser
parser = argparse.ArgumentParser(description="Generate original expert chain of thought")

# Add arguments
parser.add_argument("--base_path", type=str, required=True, help="Root directory of the dataset")
parser.add_argument("--start_id", type=int, default=0, help="Starting ID for processing")
parser.add_argument("--end_id", type=int, default=-1, help="Ending ID for processing (-1 for all remaining)")
parser.add_argument("--key", type=str, required=True, help="Google API key")

# Parse arguments
args = parser.parse_args()

# Get parameter values
base_path = args.base_path
start_id = args.start_id
end_id = args.end_id
key = args.key

print(f"Processing dataset at: {base_path}")

def encode_image(image_path):
    """Load image file"""
    image = Image.open(image_path)
    return image

def read_txt(path):
    """Read text file content"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content

# Initialize Google Gemini client
client = genai.Client(api_key=key)

# Get and sort folder list
sub_doc_name_list = os.listdir(base_path)
sub_doc_name_list.sort()    
# Filter folders based on start and end IDs
sub_doc_name_list = sub_doc_name_list[start_id: end_id] if end_id != -1 else sub_doc_name_list[start_id:]

# Process each folder
for sub_doc_name in tqdm.tqdm(sub_doc_name_list):
    try:
        # Build path
        sub_doc_path_base = os.path.join(base_path, sub_doc_name)
            
        # Set save path
        save_path = os.path.join(sub_doc_path_base, 'original_expert_cot.txt')
        # Skip if CoT file already exists
        if os.path.exists(save_path):
            print(f"Skipping existing file: {save_path}")
            continue
        
        # Load image files
        before_image = encode_image(os.path.join(sub_doc_path_base, "before.png")) if os.path.exists(os.path.join(sub_doc_path_base, "before.png")) else encode_image(os.path.join(sub_doc_path_base, "before.jpg"))
        processed_image = encode_image(os.path.join(sub_doc_path_base, "processed.png")) if os.path.exists(os.path.join(sub_doc_path_base, "processed.png")) else encode_image(os.path.join(sub_doc_path_base, "processed.jpg"))
        
        # Read configuration and user requirements
        config_txt = read_txt(os.path.join(sub_doc_path_base, "config.lua"))
        user_want = read_txt(os.path.join(sub_doc_path_base, "user_want.txt"))

        # Choose prompt template based on whether coordinate box information exists
        if os.path.exists(os.path.join(sub_doc_path_base, "box.txt")) and "portrait" in base_path:
            prompt_base = prompt_base_coor
            text_local = read_txt(os.path.join(sub_doc_path_base, "box.txt"))
            prompt_local = "\nRelative coordinates of the region of interest:" + "<box>" + text_local + "</box>"
        else:
            prompt_base = prompt_base_no
            prompt_local = ""
            
        # Build complete prompt
        prompt = prompt_base + "\nconfiguration file: " + config_txt + "\nuser requirements: " + user_want + prompt_local
        messages = [
            prompt,
            before_image,
            processed_image
        ]
        
        # Call Gemini API to generate content
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=1024
                )
            )
        )

        # Extract chain of thought part
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                result = part.text

        # Save generated CoT
        with open(save_path, 'w', encoding='utf-8') as file_cot:
            file_cot.write(result)
            
    except KeyboardInterrupt:
        print("Process interrupted by user.")
        break
    except Exception as e:
        print(f"An error occurred in generating 'cot' data. The erroneous data: {save_path}")
        print(f"Error details: {str(e)}")
