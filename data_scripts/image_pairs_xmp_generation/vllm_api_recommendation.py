import os
from openai import OpenAI
import base64
import json
import tqdm
import cv2
from prompt import *
import copy
import argparse

parser = argparse.ArgumentParser(description="Recommend presets for image editing")

# Add arguments
parser.add_argument("--base_source_path", type=str, 
                    default="./source_data", 
                    help="Root directory of source data")
parser.add_argument("--base_save_path", type=str, 
                    default="./recommendations", 
                    help="Root directory for saving results")
parser.add_argument("--base_local_path", type=str, 
                    default="./source_data/portrait_local", 
                    help="Path to local coordinate information for portrait images")
parser.add_argument("--type_source", nargs="+", type=str, default=["portrait"], 
                    help="List of source types (e.g., 'food', 'scenery', 'still_life', 'portrait')")
parser.add_argument("--start_id", type=int, default=0, help="Starting ID")
parser.add_argument("--end_id", type=int, default=-1, help="Ending ID (-1 means process to the end)")
parser.add_argument("--max_num", type=int, default=20000, help="Maximum number of images to process")

args = parser.parse_args()

base_source_path = args.base_source_path
base_save_path = args.base_save_path
base_local_path = args.base_local_path
type_source = args.type_source
max_deal_num = args.max_num
start_id = args.start_id
end_id = args.end_id


def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def list2dict(list_local):
    """Convert local detection data from list format to dictionary format."""
    return_dic = {
        "boxes": []
    }
    for i in range(len(list_local["labels"])):
        temp_dict = {
            "category": "woman" if list_local["labels"][i] == 0 else "man",
            "confidence": list_local["scores"][i],
            "coordinates": {
                "x1": list_local["bboxes"][i][0],
                "y1": list_local["bboxes"][i][1],
                "x2": list_local["bboxes"][i][2],
                "y2": list_local["bboxes"][i][3]
            }
        }
        return_dic["boxes"].append(copy.deepcopy(temp_dict))
    return return_dic


# Initialize OpenAI client
# Get API key from environment variable or use placeholder
# Set OPENAI_API_KEY environment variable or modify the api_key parameter
api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

count_sum = 0
for name_object in type_source:
    print(f"Processing type: {name_object}")
    object_name_base_path = os.path.join(base_source_path, name_object)
    object_save_base_path = os.path.join(base_save_path, name_object)
    count_sum = 0
    
    if not os.path.exists(object_save_base_path):
        os.makedirs(object_save_base_path, exist_ok=True)
    
    list_image_name = os.listdir(object_name_base_path)
    list_image_name.sort()
    list_image_name = list_image_name[start_id:end_id] if end_id != -1 else list_image_name[start_id:]
    
    for image_name in tqdm.tqdm(list_image_name):
        # Skip if already processed
        if os.path.exists(os.path.join(object_save_base_path, image_name.split(".")[0])):
            count_sum = count_sum + 1
            print(f"Skip: {count_sum}, path: {os.path.join(object_save_base_path[:-6], image_name.split('.')[0])}")
            continue
        
        # Get base path without type suffix
        base_image_path = object_name_base_path[:-6] if object_name_base_path.endswith(name_object) else object_name_base_path
        image_path = os.path.join(base_image_path, image_name.split(".")[0] + ".png")
        
        # Encode image
        before_image = encode_image(image_path)
        image_data = cv2.imread(image_path)
        
        # Prepare prompt with local detection data for portrait images
        if "portrait" in name_object:
            path_local = os.path.join(base_local_path, image_name.split(".")[0] + ".json")
            if os.path.exists(path_local):
                with open(path_local, "r", encoding="utf-8") as file:
                    local_data = json.load(file)
                if len(local_data["labels"]) == 0:
                    prompt2 = """None"""
                else:
                    local_dict_data = list2dict(local_data)
                    prompt2 = json.dumps(local_dict_data)
            else:
                prompt2 = """None"""
        else:
            prompt2 = """None"""
        
        prompt = prompt1 + prompt2 + prompt3
        
        # Call API to get recommendations
        completion = client.chat.completions.create(
            model="gpt",  # Model name, can be changed as needed
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": prompt}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{before_image}"}}
                    ]
                }
            ],
            max_tokens=10000,
        )
        
        result = json.loads(completion.model_dump_json())
        
        # Save results
        output_dir = os.path.join(object_save_base_path, image_name.split(".")[0])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cv2.imwrite(os.path.join(output_dir, image_name + ".png"), image_data)
        out_text = str(result["choices"][0]["message"]["content"])
        
        with open(os.path.join(output_dir, image_name.split(".")[0] + ".txt"), "w") as file_out:
            file_out.write(out_text)
        
        count_sum = count_sum + 1
        if count_sum >= max_deal_num:
            break
        print(f"Processed: {count_sum}")

