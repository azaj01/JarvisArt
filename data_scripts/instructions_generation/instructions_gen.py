import os
from openai import OpenAI
import base64
import time
from io import BytesIO
# from generate_data.convert2lua_local import *
import argparse
import tqdm
from PIL import Image
from google import genai
from google.genai import types

prompt_base = """
### **Role and Objective**

You are a **professional image editing intent analysis expert**. Your core task is to analyze provided image information and **infer and simulate user photo editing intentions**.

---

### **Input Information**

You will receive the following input:

- **Original Image** (Image 1)
- **Edited Version of Image** (Image 2)
- **Editing Configuration Parameters** (e.g., brightness, saturation, sharpness values)
- **Relative Coordinates of Region of Interest** (Optional, to indicate localized adjustments)

---

### **Analysis and Inference Requirements**

1. **Visual Change Analysis**: Analyze the **visual changes and adjustments** made from Image 1 to Image 2.
2. **Intent Inference**: Based on these visual changes, **infer the user's photo editing instructions or intent** for the transformation from Image 1 to Image 2.
3. **Non-Technical User Perspective**:
    - Simulate the **voice of a real customer**, describing their needs directly in **everyday English**.
    - Imagine you are the actual user, explaining your editing needs to a trusted partner.
    - **Do not mention any technical parameters, numerical values, or professional jargon** (e.g., "brightness +20", "color temperature 5500K").
    - Translate technical parameters into **approachable visual effect descriptions**.

---

### **Characteristics of Simulated User Intent**

The simulated user intent sentences you generate must meet the following criteria:

- **High Frequency and Practicality**: Use vocabulary commonly found in everyday photo editing scenarios.
- **Non-Technical**: Strictly avoid any professional editing terminology.
- **Broad Applicability**: Suitable for common daily image processing situations, including:
    - **Social Media Sharing**: Photos of people (selfies, group shots), landscapes (nature, urban), street scenes, still life (food, products), pets, and more.
    - **Mobile Photo Post-Processing**: Quick enhancements for everyday photos taken on mobile phones.
    - **Personal Mementos**: Travel photos, family photos, party photos, individual portrait close-ups, etc.

---

### **Vocabulary Selection Criteria**

When generating user intent, please refer to and **comprehensively cover** the following categories of vocabulary:

- **Mainstream App Features/Filter Names**:
    - Include and emulate names of built-in filters or features from popular photo editing apps (e.g., "vibe" from Xingtu, "vibrance" from Lightroom, "film look" from VSCO).
    - Cover common preset style names (e.g., "black and white", "vintage", "cinematic", "Japanese aesthetic", "fresh and clean").
- **Basic Visual Effect Descriptions**:
    - **Color**: Regarding color temperature ("a bit warmer", "cooler tones"), tint ("more green", "more pink"), saturation ("richer colors", "less intense colors"), vibrancy ("more vivid", "dull colors").
    - **Light & Contrast**: Regarding brightness ("brighter", "darker"), contrast ("stronger contrast", "softer look"), highlights ("dull the highlights", "brighten the highlights"), shadows ("deepen shadows", "lift shadows").
    - **Clarity & Detail**: Regarding sharpness ("sharper", "blurry details"), texture ("add texture", "reduce grain").
    - **Exposure**: Regarding overall brightness ("underexposed", "overexposed").
- **Mood and Atmosphere Expressions**:
    - **Emotional Association Words**: Vocabulary conveying specific emotions or atmospheres (e.g., "cool and lonely", "warm", "melancholy", "vibrant", "serene", "dreamy", "premium feel").
    - **Scene Atmosphere Words**: Vocabulary simulating specific scene atmospheres (e.g., "sunset evening", "rainy day blues", "summer sunshine").
- **Local Adjustment Instructions**:
    - Local editing intents that might arise from advanced operations like masks or brushes (e.g., "brighten the face", "blur the background", "make the sky bluer", "whiten the skin", "remove distractions").
- **Specific Subject Descriptions**:
    - Vocabulary directly referring to specific objects in the frame for modification (e.g., "make the eyes pop", "make the clothes more vibrant", "change the background color").
    - **General Adjectives/Adverbs**:
    - Common modifiers such as "more", "a bit", "slightly", "very", "completely", "natural", "clean", "transparent".

---

### **Intent Types and Proportions**

Based on the provided image changes, please **prioritize generating the following three categories of user intent** and allocate them according to the specified proportions:

1. **Basic Color Adjustments (34%)**:
    - **Focus**: Instructions for adjusting fundamental parameters like color temperature, tint, saturation, light/darkness, and contrast.
    - **Examples**:
        - "Make it warmer."
        - "Darken the highlights."
        - "Boost the colors."
        - "Cool down the tones."
        - "Reduce the color intensity."
        - "Brighten it up a bit."
        - "Make it a bit darker."
        - "Increase the contrast."
        - "Soften the shadows."
        - "Make the blacks deeper."
        - "Lighten the overall image."
        - "Make the colors pop more."
        - "Desaturate the image slightly."
        - "Give it a less vibrant look."
        - "Adjust the tint towards green."
        - "Make the whites brighter."
        - "Reduce the harshness of the light."
        - "Give it a more balanced exposure."
        - "Make the blues more vivid."
        - "Soften the overall color palette."
2. **Stylized Filters (33%)**:
    - **Focus**: Covering preset filter styles in popular photo editing apps or expressions that achieve similar effects.
    - **Examples**:
        - "Give it an **ins minimalist** vibe."
        - "Make it look like **Fuji film**."
        - "Add a **cyberpunk** feel."
        - "Give it a **cinematic** look."
        - "Make it feel like **old Hong Kong movies**."
        - "Convert it to **black and white**."
        - "Add a **Japanese fresh** style."
        - "Give it a **vintage** feel."
        - "Make it look like **Kodak film**."
        - "Add a **dreamy bokeh** effect."
        - "Give it a **bright and airy** feel."
        - "Make it look like a **magazine cover**."
        - "Add a **sepia** tone."
        - "Give it a **grungy** look."
        - "Make it **moody and dramatic**."
        - "Add a **pastel** effect."
        - "Give it an **oil painting** texture."
        - "Make it look like a **hand-drawn sketch**."
        - "Add a **vibrant pop** of color."
        - "Give it a **soft focus** look."
        - "Make it feel like a **warm summer day**."
        - "Add a **cool winter** ambiance."
        - "Give it a **glowing skin** effect."
        - "Make the **colors muted** and gentle."
        - "Add a **classic portrait** style."
        - "Give it a **street photography** vibe."
        - "Make it look like a **polaroid picture**."
        - "Add a **matte finish**."
        - "Give it a **cinematic blue-green** tint."
        - "Make it feel **clean and crisp**."
3. **Mood and Atmosphere (33%)**:
    - **Focus**: Color expressions based on emotional associations that convey specific moods or atmospheres.
    - **Examples**:
        - "Make it feel cold and lonely."
        - "Give it a vibrant and joyful look."
        - "Add a vintage nostalgic touch."
        - "Make it more peaceful and calm."
        - "Give it an intense and powerful feel."
        - "Make it mysterious and intriguing."
        - "Add a dreamy and ethereal touch."
        - "Give it a cozy and comfortable vibe."
        - "Make it exciting and energetic."
        - "Add a melancholic touch."
        - "Make it feel fresh and natural."
        - "Give it a serious and solemn mood."
        - "Make it playful and whimsical."
        - "Add a festive and celebratory feel."
        - "Give it a clean and minimalist look."
        - "Make it feel dark and edgy."
        - "Add a soft and delicate touch."
        - "Give it a classic and timeless appeal."
        - "Make it feel authentic and real."
        - "Add a cheerful and optimistic mood."
        - "Give it a tranquil and serene atmosphere."
        - "Make it feel dramatic and impactful."

---

### **Output Requirements and Format**

- **Overall Style Intent**: You **must include** the user's editing intent for the image's **overall style**.
- **Quantity**: You **must provide four possible user intents** for the edited image.
- **Diversity**: These four possible user intents may convey similar meanings, but their **sentence structures and vocabulary must be different**.
- **Word Limit**: Each user intent description MUST be **within 20 words**.
- **Local Adjustments Integration**: Inferred local adjustment intents should be **adaptively and naturally integrated** throughout the description text.
<prompt_local_coor>- **Strict Output Format**: Please **strictly adhere to the following JSON format** for your output:
    
    **JSON**
    
    `{
    "User Intent 1": "<first user intent>",
    "User Intent 2": "<second user intent>",
    "User Intent 3": "<third user intent>",
    "User Intent 4": "<fourth user intent>"
    }`

"""
parser = argparse.ArgumentParser(description="Generate user instructions for image editing")

parser.add_argument("--base_path", type=str, required=True, help="Root directory of the dataset")
parser.add_argument("--start_id", type=int, default=0, help="Starting ID for processing")
parser.add_argument("--end_id", type=int, default=-1, help="Ending ID for processing (-1 for all remaining)")
parser.add_argument("--key", type=str, required=True, help="Gemini API key")

args = parser.parse_args()

base_path = args.base_path
num_start = args.start_id
num_end = args.end_id

print(f"Processing range: {num_start} to {num_end}")
print(f"Base path: {base_path}")


def encode_image(image_path):
    image = Image.open(image_path)
    return image

def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content


# Add your Gemini API key here
parser.add_argument("--key", type=str, required=True, help="Gemini API key")
args = parser.parse_args()

client = genai.Client(api_key=args.key)

sub_image_name_list = os.listdir(base_path)
sub_image_name_list.sort()
sub_image_name_list = sub_image_name_list[num_start: num_end] if num_end != -1 else sub_image_name_list[num_start:]

for sub_image_name in tqdm.tqdm(sub_image_name_list):
    try:
        sub_image_base_path = os.path.join(base_path, sub_image_name)
        save_path = os.path.join(sub_image_base_path, 'user_want_2.txt')
        if os.path.exists(save_path):
            print(save_path)
            continue

        before_image = encode_image(os.path.join(sub_image_base_path, "before.png")) if os.path.exists(os.path.join(sub_image_base_path, "before.png")) else encode_image(os.path.join(sub_image_base_path, "before.jpg"))
        processed_image = encode_image(os.path.join(sub_image_base_path, "processed.png")) if os.path.exists(os.path.join(sub_image_base_path, "processed.png")) else encode_image(os.path.join(sub_image_base_path, "processed.jpg"))
        config_txt = read_txt(os.path.join(sub_image_base_path, "config.lua"))

        if os.path.exists(os.path.join(sub_image_base_path, "box.txt")) and "portrait" in base_path: 
            # prompt_base = prompt_base
            coord = read_txt(os.path.join(sub_image_base_path, "box.txt"))
            if "None" in coord:
                prompt_base = prompt_base.replace("<prompt_local_coor>", "")
                prompt_user = "The configuration details are as follows:\n" + config_txt
            else:
                prompt_local_coor = "- **Region of Interest**: If the user provides coordinates for a region of interest, ensure that ALL output identifies the category of the region of interest and includes the coordinates enclosed in `<box></box>` tags for annotation. The response should explicitly state the identified category and clearly mark the coordinates within the specified format.\n"
                prompt_base = prompt_base.replace("<prompt_local_coor>", prompt_local_coor)
                prompt_user = "Relative coordinates of the region of interest:" + coord + "\nThe configuration details are as follows:\n" + config_txt
        else:
            prompt_base = prompt_base.replace("<prompt_local_coor>", "")
            # prompt_base = prompt_base
            prompt_user = "The configuration details are as follows:\n" + config_txt
        
        messages = [
            prompt_base + prompt_user,
            before_image,
            processed_image
        ]
        count_id = 0
        while True:
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=messages,
                    config=types.GenerateContentConfig(
                        response_modalities=['TEXT']
                    )
                )
                count_id += 1
                if response.text:
                    break
                if count_id > 2:
                    break
            except KeyboardInterrupt:
                print("Program interrupted by user. Exiting...")
                exit(0)
            except Exception as e:
                time.sleep(1)
                continue
        if count_id > 10:
            print("Failed to generate response after multiple attempts. Skipping this entry.")
            continue
        result = response.text
        with open(save_path, 'w', encoding='utf-8') as file_uw:
            file_uw.write(result)
    except KeyboardInterrupt:
        print("Program interrupted by user. Exiting...")
        break
    except Exception as e:
        print("An error occurred in generating 'user want' data. The erroneous data:", sub_image_base_path)
        print("Error details:", e)