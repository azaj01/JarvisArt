
import os
import json
import base64
import random
from utils.system_prompt import SHORT_SYSTEM_PROMPT_WITH_THINKING
from utils.lua_converter import LuaConverter
from openai import OpenAI
import re
import sys
from tools.lua2lrt import lua_to_lrtemplate

def extract_json_from_answer(answer):
    """
    Extract configuration data from the answer string and convert to JSON
    Args:
        answer (str): The answer string containing configuration data
    Returns:
        list: List with exactly one configuration object
    """
    import ast
    import re
    
    def find_complete_dict(text, start_pos=0):
        """Find complete dictionary, handling nested cases"""
        brace_count = 0
        start_found = False
        start_idx = 0
        
        for i in range(start_pos, len(text)):
            char = text[i]
            if char == '{':
                if not start_found:
                    start_idx = i
                    start_found = True
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_found:
                    return text[start_idx:i+1]
        
        return None
    
    # Method 1: Find complete dictionary structure
    dict_start = answer.find('{')
    if dict_start != -1:
        complete_dict_str = find_complete_dict(answer, dict_start)
        if complete_dict_str:
            try:
                config_dict = ast.literal_eval(complete_dict_str)
                if isinstance(config_dict, dict) and len(config_dict) > 0:
                    return [config_dict]
            except Exception as e:
                print(f"⚠️ Failed to parse complete dict: {str(e)[:100]}...")
    
    # Method 2: Fallback to original method (if new method fails)
    
    # Find Python dict pattern in answer
    # Look for patterns like "{'key': value, ...}"
    dict_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    
    matches = re.findall(dict_pattern, answer, re.DOTALL)
    
    # Find the largest match
    largest_match = None
    largest_size = 0
    
    for match in matches:
        try:
            # Try to parse as Python dict using ast.literal_eval
            config_dict = ast.literal_eval(match)
            
            if isinstance(config_dict, dict) and len(config_dict) > largest_size:
                largest_match = config_dict
                largest_size = len(config_dict)
        except Exception as e:
            print(f"⚠️ Failed to parse dict: {str(e)[:50]}...")
            continue
    
    if largest_match:
        return [largest_match]
    
    print("❌ No valid configuration data found in answer")
    return []

def json_to_lua(json_data, save_folder, filepath="config.lua"):
    """
    Convert JSON data to Lua format and save to file
    Args:
        json_data (dict or str): JSON data to convert
        save_folder (str): Folder to save the Lua file
        filename (str): Filename for the Lua file
    Returns:
        tuple: (file_path, error_message)
    """
    try:
        # Ensure save folder exists
        os.makedirs(save_folder, exist_ok=True)
        
        # Parse JSON if it's a string
        if isinstance(json_data, str):
            try:
                json_obj = json.loads(json_data)
            except:
                return None, f"Error parsing JSON: Invalid JSON format"
        else:
            json_obj = json_data
        
        
        # Convert to Lua format using LuaConverter
        try:
            lua_content = LuaConverter.to_lua(json_obj)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write('return %s' % lua_content)
            
            return filepath, None
        except Exception as e:
            return None, f"Error writing Lua file: {str(e)}"
    
    except Exception as e:
        return None, f"Error in json_to_lua: {str(e)}"

# API client class, used to replace ChatModel
class APIClient:
    def __init__(self, api_endpoint, api_port, model_name="qwen2_vl", api_key="0"):
        """
        Initialize API client
        Args:
            api_endpoint (str): API server address
            api_port (int): API server port
            model_name (str): Model name
            api_key (str): API key
        """
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.api_port = api_port
        self.api_connected = False
        
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url=f"http://{api_endpoint}:{api_port}/v1",
                timeout=30.0  # Increase timeout to 30 seconds
            )
            
            # Test API connection (non-blocking)
            try:
                self.api_connected = True
            except Exception as e:
                print(f"⚠️ API connection test failed: {str(e)}")
                self.api_connected = False
                
        except Exception as e:
            print(f"❌ API client initialization failed: {str(e)}")
            print("⚠️ Program will continue to start, but API functionality is unavailable")
            self.client = None
            self.api_connected = False
    
    def chat(self, messages, system=None, images=None, **kwargs):
        """
        Chat with model via API
        
        Args:
            messages (list): Message list
            system (str): System prompt
            images (list): Image path list
            **kwargs: Other parameters
            
        Returns:
            list: List containing Response objects
        """
        try:
            # Prepare message format
            formatted_messages = []
            
            # Add system message
            if system:
                formatted_messages.append({
                    "role": "system",
                    "content": system
                })
            
            # Process user messages and images
            for msg in messages:
                if images and msg["role"] == "user":
                    # If there are images, add them to user message
                    content = []
                    
                    # Add images
                    for img_path in images:
                        # Read and encode image
                        with open(img_path, "rb") as img_file:
                            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                        
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        })
                    
                    # Add text
                    content.append({
                        "type": "text",
                        "text": msg["content"]
                    })
                    
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": content
                    })
                else:
                    # Regular text message
                    formatted_messages.append(msg)
            
            # Call API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                stream=False,
                timeout=180,  # Increase timeout
                **kwargs
            )
            
            # Create Response object for compatibility with existing code
            class Response:
                def __init__(self, text):
                    self.response_text = text
            
            # Return list containing Response object
            return [Response(response.choices[0].message.content)]
        
        except Exception as e:
            print(f"❌ API call failed: {str(e)}")
            return [Response(f"Error occurred during API call: {str(e)}")]

def get_llm_response_with_custom_prompt(image_path, user_prompt, chat_model, lua_name):
    """
    Get response from JarvisArt model using API client
    Args:
        image_path (str): Path to the input image
        user_prompt (str): User-defined prompt for analysis
        chat_model (APIClient): Instance of the API client
    Returns:
        str: Complete response from the model
    """
    try:
        # Prepare message format according to training format
        messages = [
            {
                "role": "user",
                "content": str(user_prompt)  # User prompt
            }
        ]
        
        # Prepare image input
        images = [image_path] if image_path else None
        
        # Use API client for inference
        responses = chat_model.chat(
            messages=messages,
            system=system_prompt,  # System prompt
            images=images,  # Image input
        )

        answer_match = re.search(r'<answer>(.*?)</answer>', responses[0].response_text, re.DOTALL)
        answer_content = answer_match.group(1).strip()
        json_objects = extract_json_from_answer(answer_content)
        _, error = json_to_lua(json_objects[0], os.path.dirname(image_path), lua_name)

        if error:
            return f"Error converting JSON to Lua: {error}"
        # Extract response text
        if responses and len(responses) > 0:
            return str(responses[0].response_text)
        else:
            return "Model did not return a valid response"
            
    except Exception as e:
        return f"Error occurred during inference: {str(e)}"

# Avatar file path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USER_AVATAR_PATH = os.path.join(SCRIPT_DIR, "assets", "user_avatar.svg")
AI_AVATAR_PATH = os.path.join(SCRIPT_DIR, "assets", "ai_avatar.svg")

# System prompt - pre-designed system prompt
system_prompt = SHORT_SYSTEM_PROMPT_WITH_THINKING

# Default user prompt
DEFAULT_USER_PROMPT = "I want a dreamy, romantic sunset vibe with soft, warm colors and gentle glow."



if __name__ == "__main__":
    import argparse
    
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="JarvisArt Inference Script")
    parser.add_argument("--api_endpoint", type=str, default="localhost", help="API server address")
    parser.add_argument("--api_port", type=int, default=8002, help="API server port")
    parser.add_argument("--model_name", type=str, default="qwen2_vl", help="Model name")
    parser.add_argument("--api_key", type=str, default="0", help="API key (if required)")
    parser.add_argument("--image_path", type=str, default="dataset", help="Path to the input image")
    
    args = parser.parse_args()
    

    chat_model = APIClient(
        api_endpoint=args.api_endpoint,
        api_port=args.api_port,
        model_name=args.model_name,
        api_key=args.api_key
    )

    path_list = sorted(os.listdir(args.image_path))
    for path in path_list:
            user_want_path_name = "user_want.txt"

            base_path = os.path.join(args.image_path, path)
            
            image_path = os.path.join(base_path, "before.jpg")
            save_lua_path = os.path.join(base_path, "config")
            save_text_path = os.path.join(base_path, "txt_out")
            user_prompt_path = os.path.join(base_path, user_want_path_name)

            os.makedirs(save_lua_path, exist_ok=True)
            os.makedirs(save_text_path, exist_ok=True)
            if not os.path.exists(image_path):
                image_path = os.path.join(base_path, "before.png")
            
            print(f"Processing image: {image_path}")
            

            if os.path.isfile(user_prompt_path):
                try:
                    with open(user_prompt_path, "r", encoding="utf-8") as f:
                        user_prompt = f.read()
                except Exception as e:
                    print(f"Failed to read user prompt from {user_prompt_path}: {e}")
                    sys.exit()
            else:
                user_prompt = DEFAULT_USER_PROMPT
            
            # Get model response
            response = get_llm_response_with_custom_prompt(image_path, user_prompt, chat_model, f"{save_lua_path}/output_image.lua")
            lua_to_lrtemplate(f"{save_lua_path}/out.lua")
            with open(os.path.join(save_text_path, f"response.txt"), "w", encoding="utf-8") as file:
                file.write(response)
