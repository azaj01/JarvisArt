import os
import re
import gradio as gr
import sys
import argparse
import json
import time
import hashlib
import base64
from PIL import Image

from gradio_image_annotation import image_annotator

from werkzeug.utils import secure_filename  # Add this import
from utils.system_prompt import SHORT_SYSTEM_PROMPT_WITH_THINKING
from utils.lua_converter import LuaConverter
from openai import OpenAI

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
    
    print(f"🔍 Extracting configuration from answer...")
    
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
            print(f"Found complete dictionary, length: {len(complete_dict_str)}")
            try:
                config_dict = ast.literal_eval(complete_dict_str)
                if isinstance(config_dict, dict) and len(config_dict) > 0:
                    print(f"✅ Successfully extracted configuration with {len(config_dict)} parameters")
                    print(f"📦 Config keys: {list(config_dict.keys())[:10]}...")  # Show first 10 keys
                    return [config_dict]
            except Exception as e:
                print(f"⚠️ Failed to parse complete dict: {str(e)[:100]}...")
    
    # Method 2: Fallback to original method (if new method fails)
    print("🔄 Falling back to regex pattern matching...")
    
    # Find Python dict pattern in answer
    # Look for patterns like "{'key': value, ...}"
    dict_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    
    matches = re.findall(dict_pattern, answer, re.DOTALL)
    print(f"Found {len(matches)} potential matches")
    
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
                print(f"📦 Found larger config with {len(config_dict)} parameters")
        except Exception as e:
            print(f"⚠️ Failed to parse dict: {str(e)[:50]}...")
            continue
    
    if largest_match:
        print(f"✅ Successfully extracted configuration with {len(largest_match)} parameters")
        print(f"📦 Config keys: {list(largest_match.keys())[:10]}...")  # Show first 10 keys
        return [largest_match]
    
    print("❌ No valid configuration data found in answer")
    return []

def json_to_lua(json_data, save_folder, filename="config.lua"):
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
        
        save_path = os.path.join(save_folder, filename)
        
        # Convert to Lua format using LuaConverter
        try:
            lua_content = LuaConverter.to_lua(json_obj)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write('return %s' % lua_content)
            
            return save_path, None
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
            print(f"✅ API client initialized successfully, connected to http://{api_endpoint}:{api_port}/v1")
            
            # Test API connection (non-blocking)
            try:
                print("🔍 Testing API connection...")
                response = self.client.models.list()
                available_models = [model.id for model in response.data]
                print(f"✅ API connection test successful! Available models: {available_models}")
                self.api_connected = True
            except Exception as e:
                print(f"⚠️ API connection test failed: {str(e)}")
                print("⚠️ Program will continue to start, but API functionality may not be available")
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
                timeout=60,  # Increase timeout
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
    
    def stream_chat(self, messages, system=None, images=None, **kwargs):
        """
        Stream chat with model via API
        
        Args:
            messages (list): Message list
            system (str): System prompt
            images (list): Image path list
            **kwargs: Other parameters
            
        Yields:
            str: Model generated text fragments
        """
        # Check API connection status
        if not self.api_connected or self.client is None:
            yield f"❌ API connection unavailable. Please check the following settings:\n"
            yield f"• API endpoint: {self.api_endpoint}:{self.api_port}\n"
            yield f"• Ensure SSH port forwarding is running\n"
            yield f"• Ensure API is running on remote server\n"
            yield f"• Try restarting the program"
            return
            
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
            
            # Call API (streaming)
            response_stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                stream=True,
                timeout=120,  # Increase streaming call timeout
                **kwargs
            )
            
            # Yield generated text fragments one by one
            for chunk in response_stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            print(f"❌ Streaming API call failed: {str(e)}")
            yield f"❌ API call error: {str(e)}\n"
            yield f"Please check network connection and API service status."

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="JarvisArt Gradio Demo")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="qwen2_vl",
        help="Model name to use for API calls"
    )
    parser.add_argument(
        "--api_endpoint", 
        type=str, 
        default="localhost",  
        help="API server endpoint"
    )
    parser.add_argument(
        "--api_port", 
        type=int, 
        default=8001,  
        help="API server port"
    )
    parser.add_argument(
        "--api_key", 
        type=str, 
        default="0",
        help="API key for authentication"
    )
    parser.add_argument(
        "--server_port", 
        type=int, 
        default=7880,  # Change to standard Gradio port
        help="Port for the Gradio server"
    )
    parser.add_argument(
        "--server_name", 
        type=str, 
        default="127.0.0.1",
        help="Server name/IP for the Gradio server"
    )
    parser.add_argument(
        "--share", 
        action="store_true",
        help="Enable public sharing via Gradio tunnel (creates public URL)"
    )
    return parser.parse_args()

# Get command line arguments
args = parse_args()

# Avatar file path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USER_AVATAR_PATH = os.path.join(SCRIPT_DIR, "assets", "user_avatar.svg")
AI_AVATAR_PATH = os.path.join(SCRIPT_DIR, "assets", "ai_avatar.svg")

# System prompt - pre-designed system prompt
system_prompt = SHORT_SYSTEM_PROMPT_WITH_THINKING

# Default user prompt
default_user_prompt = "I want a dreamy, romantic sunset vibe with soft, warm colors and gentle glow."

# Initialize API client
print(f"Connecting to API server {args.api_endpoint}:{args.api_port}...")
print("🔧 Connection configuration:")
print(f"   📍 API Address: {args.api_endpoint}")
print(f"   🚪 API Port: {args.api_port}")
print(f"   🤖 Model Name: {args.model_name}")
print(f"   🔑 API Key: {'Set' if args.api_key != '0' else 'Default'}")

chat_model = APIClient(
    api_endpoint=args.api_endpoint,
    api_port=args.api_port,
    model_name=args.model_name,
    api_key=args.api_key
)

# Check API connection status
if hasattr(chat_model, 'api_connected') and chat_model.api_connected:
    print("✅ Connecting to API server...")
else:
    print("⚠️ API client initialized, but connection may have issues")
    print("💡 Common solutions:")
    print(f"   1. Check if API service is running at {args.api_endpoint}:{args.api_port}")
    print("   2. Check firewall settings")
    print("   3. Try using a different port (e.g., --api_port 8001)")
    print("   4. If API is on a remote server, check SSH port forwarding")
    print("   5. Use command line arguments: --api_endpoint <address> --api_port <port>")
print("="*60)


def parse_llm_response(response):
    """
    Parse the LLM response to extract reason and answer sections
    
    Args:
        response (str): The raw response from the LLM
        
    Returns:
        tuple: (reason, answer) extracted from the response
    """
    # Ensure response is a string
    response = str(response)
    
    # Try to parse <think> and <answer> tags (corresponding to SYSTEM_PROMPT_WITH_THINKING)
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    
    if think_match and answer_match:
        thinking = think_match.group(1).strip()
        answer = answer_match.group(1).strip()
        return thinking, answer
    
    
    # If nothing found, return the entire response as the answer with empty thinking
    return "AI is thinking...", response

def extract_models_from_answer(answer):
    """
    Extract model names from the answer string using regex
    
    Args:
        answer (str): The answer string containing model recommendations
        
    Returns:
        list: List of extracted model names
    """
    # Pattern to match [type:xxx]:(model:xxx)
    pattern = r'\[type:[^\]]+\]:\(model:([^)]+)\)'
    models = re.findall(pattern, answer)
    return models

def get_llm_response_with_custom_prompt(image_path, user_prompt):
    """
    Get response from JarvisArt model using API client
    
    Args:
        image_path (str): Path to the input image
        user_prompt (str): User-defined prompt for analysis
        
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

        # Extract response text
        if responses and len(responses) > 0:
            return str(responses[0].response_text)
        else:
            return "Model did not return a valid response"
            
    except Exception as e:
        return f"Error occurred during inference: {str(e)}"

def get_llm_response_with_custom_prompt_stream(image_path, user_prompt, max_new_tokens=10240, top_k=50, top_p=0.8, temperature=0.7):
    """
    Get streaming response from JarvisArt model using API client
    
    Args:
        image_path (str): Path to the input image
        user_prompt (str): User-defined prompt for analysis
        max_new_tokens (int): Maximum number of new tokens to generate
        top_k (int): Top-k sampling parameter
        top_p (float): Top-p (nucleus) sampling parameter
        temperature (float): Temperature for sampling
        
    Yields:
        str: Streaming response tokens from the model
    """
    global chat_model
    
    if chat_model is None:
        yield "❌ API client not initialized. Please restart the program."
        return
        
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
        
        # Prepare generation parameters
        generation_kwargs = {
            "max_tokens": max_new_tokens,
            "temperature": temperature
        }
        
        # Only add top_k and top_p parameters when API supports them
        if hasattr(chat_model, 'api_connected') and chat_model.api_connected:
            # Note: OpenAI API may not support top_k parameter, only use temperature and top_p
            generation_kwargs["top_p"] = top_p
        
        # Use API client for streaming inference
        for new_token in chat_model.stream_chat(
            messages=messages,
            system=system_prompt,  # System prompt
            images=images,  # Image input
            **generation_kwargs  # Pass generation parameters
        ):
            yield new_token
            
    except Exception as e:
        yield f"❌ Error during inference: {str(e)}"

def process_upload(file):
    return file

def compact_text(text):
    """
    Process the text, remove extra line breaks and spaces, and make the text more compact
    
    Args:
        text (str): Input text

    Returns:
        str: Processed text
    """
    # Remove extra line breaks
    text = re.sub(r'\n\s*\n', '\n', text)
    # Remove leading whitespace
    text = re.sub(r'\n\s+', '\n', text)
    # Remove extra spaces
    text = re.sub(r' {2,}', ' ', text)
    return text

def get_box_coordinates(annotated_image_dict, prompt_original):
    """
    Processes the output from the image_annotator to extract
    and format the bounding box coordinates.
    """
    global local_dict
    if annotated_image_dict and annotated_image_dict["boxes"]:
        # Get the last drawn box
        input_image = annotated_image_dict["image"]
        pil_image = Image.open(input_image)
        last_box = annotated_image_dict["boxes"][-1]
        width, height = pil_image.width, pil_image.height

        xmin = last_box["xmin"] / width
        ymin = last_box["ymin"] / height
        xmax = last_box["xmax"] / width
        ymax = last_box["ymax"] / height
        
        local_dict[input_image] = [xmin, ymin, xmax, ymax]
        # Format the coordinates into a string
        
        return str([xmin, ymin, xmax, ymax]), " In the region <box></box>, xxx"
    return "No box drawn", prompt_original

def process_analysis_pipeline_stream(image_dict, user_prompt, max_new_tokens, top_k, top_p, temperature):
    """
    Main analysis pipeline with streaming output, modern chat interface style, and image display support
    
    Args:
        image (str): Path to the input image
        user_prompt (str): User-defined prompt for analysis
        max_new_tokens (int): Maximum number of new tokens to generate
        top_k (int): Top-k sampling parameter
        top_p (float): Top-p (nucleus) sampling parameter
        temperature (float): Temperature for sampling
        
    Yields:
        list: Updated chat_history for Gradio UI updates (messages format)
    """
    if image_dict is None:
        yield [
            {"role": "user", "content": "Please upload an image first! 📸"},
            {"role": "assistant", "content": "I need an image to analyze before I can provide editing recommendations."}
        ]
        return
    image = image_dict['image']
    if not user_prompt.strip():
        user_prompt = default_user_prompt
    elif len(local_dict) > 0 and local_dict[image][0] != local_dict[image][2]:
        user_prompt = user_prompt.replace('<box></box>', f'<box>{str(local_dict[image])}</box>')

    
    try:
        # Initialize chat history with user message including image
        chat_history = []
        
        # Create user message with image and instructions - using messages format
        user_message_text = f"**Instructions:** {user_prompt}".replace('<box>', f'(').replace('</box>', f')')
        
        # Add user message with image
        if image_dict:
            # For messages format, we need to handle images differently
            # First add the image
            chat_history.append({
                "role": "user", 
                "content": {
                    "path": image,
                    "mime_type": "image/jpeg"
                }
            })
            # Then add text message
            chat_history.append({
                "role": "user",
                "content": user_message_text
            })
        else:
            chat_history.append({
                "role": "user",
                "content": user_message_text
            })
        yield chat_history
        
        # JarvisArt starts responding
        chat_history.append({
            "role": "assistant",
            "content": "<div style='margin:0;padding:0'>🎨 <strong style='margin:0;padding:0'>JarvisArt is analyzing your image...</strong><br/><em>Please wait while I examine the details and understand your vision.</em></div>"
        })
        ai_message_index = len(chat_history) - 1  # Record AI message index position
        recommendations_index = None  # Initialize recommendations message index
        yield chat_history
        
        # Get streaming response
        full_response = ""
        token_count = 0
        update_frequency = 8  # Reduce update frequency for smoother experience
        
        # Stage marker
        stage = "starting"  # starting, thinking, answer, completed
        answer_completed = False  # Flag to track if answer is completed
        
        for new_token in get_llm_response_with_custom_prompt_stream(
            image, user_prompt, max_new_tokens, top_k, top_p, temperature
        ):
            full_response += new_token
            token_count += 1
            
            # Detect thinking stage
            if "<think>" in full_response and stage == "starting":
                stage = "thinking"
                chat_history[ai_message_index] = {
                    "role": "assistant",
                    "content": "💭 **Thinking Process**\n*Analyzing image characteristics and understanding your creative vision...*"
                }
                yield chat_history
                continue
            
            # Thinking completed
            if "</think>" in full_response and stage == "thinking":
                stage = "between"
                think_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
                if think_match:
                    thinking_content = think_match.group(1).strip()
                    # Use the compact_text function to process text
                    thinking_content = compact_text(thinking_content).replace('<box>', f'(').replace('</box>', f')')
                    # Use special formatting to force eliminate spacing
                    formatted_thinking = f"<div style='margin:0;padding:0'>💭 <strong style='margin:0;padding:0'>Thinking</strong><div style='margin:0;padding:0'>{thinking_content}</div></div>"
                    chat_history[ai_message_index] = {
                        "role": "assistant",
                        "content": formatted_thinking
                    }
                    yield chat_history
                continue
            
            # Detect answer stage
            if "<answer>" in full_response and stage in ["between", "thinking"]:
                stage = "answer"
                # Use special formatting to force eliminate spacing
                initial_recommendations = "<div style='margin:0;padding:0;margin-top:-30px'>✨ <strong style='margin:0;padding:0'>Professional Editing Recommendations</strong><div style='margin:0;padding:0'>*Generating personalized editing suggestions...*</div></div>"
                chat_history.append({
                    "role": "assistant",
                    "content": initial_recommendations
                })
                recommendations_index = len(chat_history) - 1  # Record recommendations message index
                yield chat_history
                continue
            
            # Answer completed
            if "</answer>" in full_response and stage == "answer" and not answer_completed:
                stage = "completed"
                answer_completed = True
                answer_match = re.search(r'<answer>(.*?)</answer>', full_response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # Use the compact_text function to process text
                    answer_content = compact_text(answer_content)

                    # Use special formatting to force eliminate spacing
                    formatted_answer = f"<div style='margin:0;padding:0;margin-top:-30px'>✨ <strong style='margin:0;padding:0'>Professional Editing Recommendations</strong><div style='margin:0;padding:0'>{answer_content}</div></div>"
                    
                    chat_history[recommendations_index] = {
                        "role": "assistant",
                        "content": formatted_answer
                    }
                    yield chat_history
                # Don't break here - continue to Final completion for JSON extraction
            
            # Real-time content updates (reduced frequency) - only if answer not completed
            if token_count % update_frequency == 0 and not answer_completed:
                if stage == "thinking":
                    current_thinking = full_response.split("<think>")[-1].replace("</think>", "").strip()
                    if current_thinking and len(current_thinking) > 20:  # Avoid displaying too short content
                        # Use the compact_text function to process text
                        current_thinking = compact_text(current_thinking)
                        # Use special formatting to force eliminate spacing
                        formatted_thinking = f"<div style='margin:0;padding:0'>💭 <strong style='margin:0;padding:0'>Thinking</strong><div style='margin:0;padding:0'>{current_thinking}...<br/><em>Still analyzing...</em></div></div>"
                        chat_history[ai_message_index] = {
                            "role": "assistant",
                            "content": formatted_thinking
                        }
                        yield chat_history
                
                elif stage == "answer":
                    current_answer = full_response.split("<answer>")[-1].replace("</answer>", "").strip()
                    if current_answer and len(current_answer) > 30:  # Avoid displaying too short content
                        # Use the compact_text function to process text
                        current_answer = compact_text(current_answer)
                        # Use special formatting to force eliminate spacing
                        formatted_answer = f"<div style='margin:0;padding:0;margin-top:-30px'>✨ <strong style='margin:0;padding:0'>JarvisArt Recommendations</strong><div style='margin:0;padding:0'>{current_answer}...<br/><em>Generating more suggestions...</em></div></div>"
                        if recommendations_index is not None:
                            chat_history[recommendations_index] = {
                                "role": "assistant",
                                "content": formatted_answer
                            }
                        else:
                            chat_history.append({
                                "role": "assistant",
                                "content": formatted_answer
                            })
                            recommendations_index = len(chat_history) - 1
                        yield chat_history
        
        # Final completion
        if stage == "completed":
            # Analysis is complete, now process and save lua files
            print(f"🔍 Debug: Final completion stage reached")
            answer_match = re.search(r'<answer>(.*?)</answer>', full_response, re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
                print(f"🔍 Debug: Extracted answer content (first 200 chars): {answer_content[:200]}...")
                
                # Extract JSON objects from the answer
                json_objects = extract_json_from_answer(answer_content)
                print(f"🔍 Debug: Found {len(json_objects)} JSON objects")
                
                # Save JSON objects as lua files
                if json_objects:
                    print(f"🔍 Debug: Processing {len(json_objects)} JSON objects for conversion")
                    conversion_index = None
                    chat_history.append({
                        "role": "assistant",
                        "content": "<div style='margin:0;padding:0;margin-top:-20px'>⚙️ <strong style='margin:0;padding:0'>Lightroom Configuration Converting...</strong><br/><em>Converting editing parameters to Lightroom-compatible format...</em></div>"
                    })
                    conversion_index = len(chat_history) - 1
                    yield chat_history
                    
                    # Create lua_results folder in the same directory as this script
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    results_dir = os.path.join(script_dir, "results")
                    os.makedirs(results_dir, exist_ok=True)
                    
                    # Generate timestamp for unique session folder name
                    timestamp = int(time.time())
                    session_folder_name = f"example_{timestamp}"
                    session_dir = os.path.join(results_dir, session_folder_name)
                    os.makedirs(session_dir, exist_ok=True)
                    
                    # Copy the uploaded image to the session folder
                    import shutil
                    # Use secure_filename and hash to generate unique original image filename, avoiding conflicts with processed images
                    original_filename = secure_filename(os.path.basename(image))
                    file_hash = hashlib.md5(f"{original_filename}_{time.time()}".encode()).hexdigest()
                    
                    # Keep original extension
                    file_ext = os.path.splitext(original_filename)[1] or '.jpg'
                    unique_original_filename = f"original_{file_hash}{file_ext}"
                    
                    image_dest_path = os.path.join(session_dir, unique_original_filename)
                    shutil.copy2(image, image_dest_path)
                    
                    # Save the full model response to a text file
                    response_file_path = os.path.join(session_dir, "full_response.txt")
                    with open(response_file_path, "w", encoding="utf-8") as f:
                        f.write(full_response)
                    
                    # Save user prompt to a text file
                    prompt_file_path = os.path.join(session_dir, "user_prompt.txt")
                    with open(prompt_file_path, "w", encoding="utf-8") as f:
                        f.write(user_prompt)
                    
                    saved_files = []
                    for i, json_obj in enumerate(json_objects):
                        filename = f"config_{i+1}.lua"
                        lua_path, error = json_to_lua(json_obj, session_dir, filename)
                        
                        if lua_path:
                            saved_files.append(lua_path)
                            print(f"✅ Saved Lua config: {lua_path}")
                        else:
                            print(f"❌ Failed to save Lua config {i+1}: {error}")
                    
                    
                    # Update file save notification
                    if saved_files:
                        save_notification = "<div style='margin:0;padding:0;margin-top:-20px'>"
                        save_notification += "✅ <strong style='margin:0;padding:0'>Files saved successfully!</strong><br/>"
                        save_notification += "📁 <strong>Save location:</strong> <code>results/" + session_folder_name + "/</code><br/>"
                        save_notification += "📄 <strong>Generated files:</strong><br/>"
                        save_notification += "   • Original image: <code>" + unique_original_filename + "</code><br/>"
                        save_notification += "   • Full response: <code>full_response.txt</code><br/>"
                        save_notification += "   • User prompt: <code>user_prompt.txt</code><br/>"
                        save_notification += "   • Config files: " + str(len(saved_files)) + " files"
                        
                        save_notification += "<br/><strong>Config files:</strong>"
                        for i, file_path in enumerate(saved_files):
                            filename = os.path.basename(file_path)
                            save_notification += "<br/>   • <code>" + filename + "</code>"
                        
                        save_notification += "<br/>💡 <strong>Usage:</strong> Import <code>.lua</code> files into compatible applications to apply recommended settings"
                        save_notification += "</div>"
                        
                        # Use the compact_text function to process text
                        save_notification = compact_text(save_notification)
                        
                        # Update conversion message
                        if conversion_index is not None:
                            chat_history[conversion_index] = {
                                "role": "assistant",
                                "content": save_notification
                            }
                    else:
                        # Show conversion failed message
                        if conversion_index is not None:
                            chat_history[conversion_index] = {
                                "role": "assistant",
                                "content": "<div style='margin:0;padding:0;margin-top:-20px'>❌ <strong style='margin:0;padding:0'>Lightroom config conversion failed</strong><br/><em>No valid configuration data found in recommendations.</em></div>"
                            }
                else:
                    print(f"🔍 Debug: No JSON objects found, adding debug message to chat")
                    # Add debug message to show what was found
                    debug_msg = "<div style='margin:0;padding:0;margin-top:-20px'>"
                    debug_msg += "🔍 <strong style='margin:0;padding:0'>Debug Information</strong><br/>"
                    debug_msg += "<strong>Answer Content Preview:</strong><br/><pre style='margin:0;padding:4px'>" + answer_content[:500] + "...</pre><br/>"
                    debug_msg += "<strong>Extraction Attempted:</strong> No valid JSON objects found in the recommendations."
                    debug_msg += "</div>"
                    
                    # Use the compact_text function to process text
                    debug_msg = compact_text(debug_msg)
                    
                    chat_history.append({
                        "role": "assistant",
                        "content": debug_msg
                    })
            else:
                print(f"🔍 Debug: No answer match found in full_response")
        else:
            # If not ended normally, try to parse and format final response
            print(f"🔍 Debug: Non-normal completion, stage: {stage}")
            think_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', full_response, re.DOTALL)
            
            if think_match:
                thinking_content = think_match.group(1).strip()
                formatted_thinking = f"💭 **Thinking**\n{thinking_content}"
                chat_history[ai_message_index] = {
                    "role": "assistant",
                    "content": formatted_thinking
                }
            
            if answer_match:
                answer_content = answer_match.group(1).strip()
                formatted_answer = f"✨ **Professional Editing Recommendations**\n{answer_content}"
                if recommendations_index is not None:
                    chat_history[recommendations_index] = {
                        "role": "assistant",
                        "content": formatted_answer
                    }
                else:
                    chat_history.append({
                        "role": "assistant",
                        "content": formatted_answer
                    })
                
                # Extract and save JSON objects from answer even if not completed normally
                json_objects = extract_json_from_answer(answer_content)
                print(f"🔍 Debug: Non-normal completion found {len(json_objects)} JSON objects")
                
                if json_objects:
                    # Show Lightroom configuration conversion in progress
                    conversion_index = None
                    chat_history.append({
                        "role": "assistant",
                        "content": "<div style='margin:0;padding:0;margin-top:-20px'>⚙️ <strong style='margin:0;padding:0'>Lightroom Configuration Converting...</strong><br/><em>Converting editing parameters to Lightroom-compatible format...</em></div>"
                    })
                    conversion_index = len(chat_history) - 1
                    yield chat_history

                    # Same processing logic... (omitting repetitive code here for brevity)
                    # [Continue processing logic, format as above]

        yield chat_history
            
    except Exception as e:
        error_msg = f"❌ **Oops! Something went wrong**\n\n```\nError: {str(e)}\n```\n\n💡 **Try again with:**\n- A different image format\n- A simpler description\n- Refreshing the page"
        chat_history = [
            {"role": "user", "content": "Image analysis request"},
            {"role": "assistant", "content": error_msg}
        ]
        yield chat_history

# Create Gradio interface
def create_interface():
    """
    Create and configure the modern chat-style Gradio web interface
    
    Returns:
        gr.Blocks: Configured Gradio interface with chat-style layout
    """
    # Custom CSS styles mimicking modern chat interfaces
    custom_css = """
    /* Global styles */
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Header styles - clean white background */
    .app-header {
        background: white;
        color: #1f2937;
        padding: 0.1rem 2rem;
        border-radius: 12px 12px 0 0;
        margin-bottom: 0 !important;
        text-align: center;
        border: 1px solid #e5e7eb;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .app-title {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        font-weight: 600;
        margin: 0;
        color: #1f2937;
    }
    .centered-image {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: auto !important;
    }
    .centered-image img {
        max-width: 100% !important;
        max-height: 100% !important;
        object-fit: contain;
    }
    .jarvis_image_container {
        display: flex !important;
        justify-content: center !important;
        background: white;
        align-items: center !important;
    }
    .tight-container {
        padding: 0 !important;
        margin: 0 !important;
        height: fit-content !important;  /* Container height adapts to content */
    }
    /* Main chat area - ensure borders are visible and connect seamlessly with header */
    .chat-container {
        height: 600px;
        border: 1px solid #e5e7eb !important;
        border-top: 1px solid #e5e7eb !important;
        border-radius: 0 0 12px 12px;
        background: #ffffff !important;
        margin-top: -1px !important;
        margin-bottom: 0 !important;
        overflow-y: auto !important;
        max-height: none !important;
    }
    
    /* Chat message styles */
    .chatbot {
        border: 1px solid #e5e7eb !important;
        border-top: none !important;
        background: #ffffff !important;
        border-radius: 0 0 12px 12px !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        overflow-y: auto !important;
        max-height: none !important;
    }
    
    /* Reduce spacing between messages */
    .chatbot .message {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Reduce spacing between message containers */
    .chatbot .block {
        gap: 0 !important;
        margin-bottom: -8px !important;
    }

    /* Reduce spacing between thinking and recommendation messages */
    .chatbot .message-wrap {
        margin-bottom: -8px !important;
    }

    /* Reduce spacing between adjacent messages */
    .chatbot .message + .message {
        margin-top: -10px !important;
    }

    /* Reduce spacing between bot messages */
    .chatbot .bot-message {
        margin-top: -5px !important;
    }

    /* Special control over spacing between thinking and recommendation messages */
    .chatbot .message:nth-child(odd) + .message:nth-child(even) {
        margin-top: -15px !important;
    }
    
    /* Ensure message content area fully expands */
    .chatbot .message .content {
        max-height: none !important;
        overflow: visible !important;
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        line-height: 1.2 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Remove any height limitations */
    .chatbot .message-bubble {
        max-height: none !important;
        overflow: visible !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Ensure entire chat area can scroll, but individual messages fully expand */
    .chatbot .messages-container {
        max-height: 550px !important;
        overflow-y: auto !important;
    }
    
    /* Image styles in user messages */
    .chatbot .message img {
        max-width: 300px !important;
        max-height: 400px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        margin-bottom: 4px !important;
        margin-top: 4px !important;
        object-fit: cover !important;
        display: block !important;
    }
    
    /* Processed image styles - larger display for results */
    .chatbot .message img[src*="lr_processed"] {
        max-width: 450px !important;
        max-height: 500px !important;
        border: 2px solid #e5e7eb !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15) !important;
        margin: 8px 0 !important;
        object-fit: contain !important;
    }
    
    /* All processed result images - enhance visibility */
    .chatbot .message img[src*="results/"] {
        max-width: 500px !important;
        max-height: 600px !important;
        border: 3px solid #3b82f6 !important;
        border-radius: 20px !important;
        box-shadow: 0 10px 20px rgba(59, 130, 246, 0.2) !important;
        margin: 12px auto !important;
        object-fit: contain !important;
        display: block !important;
        background: white !important;
        padding: 4px !important;
    }
    
    /* Enhanced display for assistant images */
    .chatbot .bot-message img {
        border: 2px solid #059669 !important;
        border-radius: 16px !important;
        box-shadow: 0 6px 12px rgba(5, 150, 105, 0.15) !important;
        max-width: 100% !important;
        height: auto !important;
    }
    
    /* Processed image container styling */
    .chatbot .message .processed-image-container {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        margin: 8px 0 !important;
        padding: 8px !important;
        background: #f8fafc !important;
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Multimodal message container */
    .chatbot .message-content {
        display: flex !important;
        flex-direction: column !important;
        gap: 2px !important;
    }
    
    .chatbot .message .multimodal-content {
        display: flex !important;
        flex-direction: column !important;
        gap: 4px !important;
    }
    
    .chatbot .message .multimodal-content img {
        max-width: 100% !important;
        height: auto !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        margin: 2px 0 !important;
    }
    
    .chatbot .message .multimodal-content .text-content {
        font-size: 0.95rem !important;
        line-height: 1.2 !important;
        color: inherit !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* User message bubble styles */
    .chatbot .user-message {
        background: #f3f4f6 !important;
        color: #1f2937 !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 8px 12px !important;
        margin: 4px 0 !important;
        max-width: 80% !important;
        margin-left: auto !important;
        border: 1px solid #e5e7eb !important;
    }
    
    /* AI message bubble styles */
    .chatbot .bot-message {
        background: white !important;
        color: #374151 !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 8px 12px !important;
        margin: 2px 0 !important;
        max-width: 80% !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        margin-top: -5px !important;
        margin-bottom: 0 !important;
    }

    /* Special styles for titles */
    .chatbot .message .content strong {
        display: inline-block !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }

    /* Special styles for content following Thinking and Recommendations titles */
    .chatbot .message .content strong + br {
        display: none !important;
    }

    /* Further reduce spacing between thinking and recommendation messages */
    .chatbot .message:nth-last-child(n+2) {
        margin-bottom: -12px !important;
    }
    
    /* Override paragraph spacing in messages */
    .chatbot .message p {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Override list spacing in messages */
    .chatbot .message ul, 
    .chatbot .message ol {
        margin-top: 2px !important;
        margin-bottom: 2px !important;
        padding-left: 20px !important;
    }
    
    .chatbot .message li {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.2 !important;
    }
    
    /* Reduce spacing between emoji and title */
    .chatbot .message .content span {
        margin-right: 0 !important;
        padding-right: 0 !important;
    }
    
    /* Reduce spacing between title and content */
    .chatbot .message .content strong + span {
        margin-left: 0 !important;
        padding-left: 0 !important;
    }
    
    /* Input area styles */
    .input-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
        border: 1px solid #e5e7eb;
    }
    
    .image-upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        background: #f8fafc;
    }
    
    .image-upload-area:hover {
        border-color: #667eea;
        background: #f1f5f9;
    }
    
    /* Ensure the image upload component fits within the upload area */
    .image-upload-area .gradio-image {
        border: none !important;
        background: transparent !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .image-upload-area .image-container {
        border: none !important;
        background: transparent !important;
    }
      
    /* Remove borders from all child components in upload area */
    .image-upload-area > * {
        border: none !important;
    }
    
    .image-upload-area .gradio-group {
        border: none !important;
        background: transparent !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .image-upload-area .gradio-column {
        border: none !important;
        background: transparent !important;
    }
    
    /* Style the upload drop zone */
    .image-upload-area .upload-zone {
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        background: white !important;
        margin-top: 0.5rem !important;
    }
    
    /* Simplified button styles */
    .simple-button {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        color: #475569 !important;
        font-weight: 500 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    
    .simple-button:hover {
        background: #e2e8f0 !important;
        border-color: #cbd5e1 !important;
        color: #334155 !important;
    }
    
    .primary-simple-button {
        background: #667eea !important;
        border: 1px solid #667eea !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 500 !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.2s ease !important;
    }
    
    .primary-simple-button:hover {
        background: #5a67d8 !important;
        border-color: #5a67d8 !important;
    }
    
    /* Text input styles */
    .prompt-input {
        border-radius: 8px !important;
        border: 1px solid #e1e5e9 !important;
        padding: 0.75rem !important;
        font-size: 0.95rem !important;
    }
    
    .prompt-input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* System prompt area */
    .system-prompt-section {
        margin-top: 2rem;
        padding: 1rem;
        background: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e1e5e9;
    }
    
    /* Example button styles */
    .example-buttons button {
        background: #f1f5f9 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        font-size: 0.85rem !important;
        color: #475569 !important;
        transition: all 0.2s ease !important;
    }
    
    .example-buttons button:hover {
        background: #e2e8f0 !important;
        border-color: #667eea !important;
        color: #667eea !important;
    }
    
    /* Welcome screen styles */
    .welcome-message {
        text-align: center;
        padding: 1rem;
        color: #64748b;
    }
    
    .welcome-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.5rem;
    }
    
    .welcome-subtitle {
        font-size: 1rem;
        margin-bottom: 1rem;
        line-height: 1.4;
    }
    
    .welcome-features {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 0.5rem;
        margin-top: 1rem;
    }
    
    .welcome-feature {
        background: white;
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    .welcome-feature-icon {
        font-size: 1.5rem;
        margin-bottom: 0.25rem;
    }
    
    .welcome-feature-title {
        font-weight: 600;
        margin-bottom: 0.15rem;
        color: #334155;
    }
    
    .welcome-feature-desc {
        font-size: 0.85rem;
        color: #64748b;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .gradio-container {
            margin: 0 1rem !important;
        }
        
        .app-header {
            padding: 0.75rem 1rem !important;
        }
        
        .app-title {
            font-size: 1.4rem !important;
        }
        
        .input-section {
            padding: 1rem !important;
        }
        
        .chatbot .message img {
            max-width: 250px !important;
            max-height: 300px !important;
        }
        
        .chatbot .user-message,
        .chatbot .bot-message {
            max-width: 90% !important;
        }
        
        .welcome-features {
            grid-template-columns: 1fr !important;
        }
    }
    
    /* Remove gap between header and chatbot */
    .app-header + .gradio-row {
        margin-top: 0 !important;
        gap: 0 !important;
    }
    
    /* Ensure no gaps between Row components */
    .gradio-row {
        gap: 0 !important;
        margin: 0 !important;
    }
    
    /* Remove unnecessary margins and padding */
    .block {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Ensure header row has no gaps */
    .gradio-container > .gradio-row:first-child {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Use negative margin to force components closer - most aggressive approach */
    .chat-container {
        margin-top: -30px !important;
    }

    /* Ensure row containing chatbot has no top margin */
    .gradio-container > .gradio-row:nth-child(2) {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Enforce message interval constraints */
    .chatbot .message-wrap {
        margin-bottom: -30px !important;
        position: relative !important;
        z-index: 1 !important;
    }

    /* Enforce reduced spacing between adjacent messages */
    .chatbot .message + .message {
        margin-top: -30px !important;
    }

    /* Special constraints for bot messages */
    .chatbot .bot-message {
        background: white !important;
        color: #374151 !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 8px 12px !important;
        margin: 0 !important;
        max-width: 80% !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        margin-top: -15px !important;
        margin-bottom: -10px !important;
        position: relative !important;
        z-index: 0 !important;
    }

    /* Enforce removal of whitespace between messages */
    .chatbot .block {
        gap: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Enforce message container constraints */
    .chatbot .messages-container {
        display: flex !important;
        flex-direction: column !important;
        gap: 0 !important;
    }

    /* Enforce removal of spacing between messages */
    .chatbot .message {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Enforce removal of spacing between title and content */
    .chatbot .message .content {
        display: inline-block !important;
    }

    /* Special styles for titles */
    .chatbot .message .content strong {
        display: inline-block !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Extreme reduction of spacing between thinking and recommendation messages */
    .chatbot .message:nth-child(n+2):nth-child(-n+3) {
        margin-top: -50px !important;
    }

    /* Use transform to move element position */
    .chatbot .message:nth-child(3) {
        transform: translateY(-30px) !important;
    }

    /* Use negative margin to force elements overlap */
    .chatbot .message:nth-child(2) {
        margin-bottom: -40px !important;
    }

    /* Use absolute positioning to control element position */
    .chatbot .messages-container {
        position: relative !important;
    }

    /* Enforce removal of any potential spacing */
    .chatbot * {
        margin-bottom: 0 !important;
    }
    """
    
    with gr.Blocks(
        title="JarvisArt", 
        css=custom_css
    ) as demo:
        
        # Header area
        with gr.Row(elem_classes="app-header"):
            gr.HTML("""
                <div class="app-title">
                    <svg t="1748332876263" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" 
                         width="24" height="24" style="vertical-align: middle; margin-right: 8px;">
                        <path d="M938.56 511.36a238.72 238.72 0 0 1-238.72 238.72H659.2c-35.84 0-70.08-14.72-95.04-40.32-2.24-2.24-4.16-4.48-7.04-7.04l-6.08-7.36a130.176 130.176 0 0 0-65.28-42.24 131.456 131.456 0 0 0-149.12 58.88L304 767.68c-9.28 15.68-22.4 28.16-38.08 37.12a104.48 104.48 0 0 1-113.28-7.04c-17.6-12.8-31.04-31.68-37.76-53.44-19.2-64-28.8-130.24-28.8-196.8v-35.84c0-117.76 47.68-224.32 124.8-301.44s183.68-124.8 301.44-124.8c104.96 0 201.28 38.08 275.52 101.12l-47.36 73.6-74.24 114.88-44.48 68.48c-51.2 0.64-92.8 42.24-92.8 93.76 0 13.44 2.88 26.24 8 37.76 12.48 28.16 38.4 49.28 69.76 54.4 5.12 0.96 10.56 1.6 16 1.6 51.84 0 93.76-41.92 93.76-93.76 0-25.28-9.92-48.32-26.24-64.96l133.44-251.2c71.36 75.84 114.88 178.24 114.88 290.24z" fill="#E2CCA8"></path>
                        <path d="M564.16 709.76l-11.52 21.76c-10.24 19.2-31.36 28.48-51.52 24.96-3.52-3.52-7.36-6.08-11.52-8.64-5.44-3.2-10.88-5.44-16.32-7.04-14.72-16-17.28-40.64-4.8-59.52l18.24-28.16a133.12 133.12 0 0 1 65.28 42.24l6.08 7.36 6.08 7.04z" fill="#AA571B"></path>
                        <path d="M517.44 852.8c-21.12 36.8-102.08 108.16-138.88 87.04-36.8-21.12-15.04-127.04 6.08-163.84a76.832 76.832 0 0 1 104.64-28.16c4.16 2.24 8 5.12 11.52 8.64 27.84 23.04 35.52 63.68 16.64 96.32z" fill="#F2C05C"></path>
                        <path d="M231.04 639.36m-59.52 0a59.52 59.52 0 1 0 119.04 0 59.52 59.52 0 1 0-119.04 0Z" fill="#61A838"></path>
                        <path d="M265.28 417.6m-76.8 0a76.8 76.8 0 1 0 153.6 0 76.8 76.8 0 1 0-153.6 0Z" fill="#FF737D"></path>
                        <path d="M435.52 247.36m-76.8 0a76.8 76.8 0 1 0 153.6 0 76.8 76.8 0 1 0-153.6 0Z" fill="#FAAD14"></path>
                        <path d="M740.48 259.84l-74.24 114.88a76.768 76.768 0 0 1-68.8-76.48c0-42.56 34.24-76.8 76.8-76.8 28.16 0.32 53.12 15.68 66.24 38.4z" fill="#40A9FF"></path>
                    </svg>
                    <span>JarvisArt</span>
                </div>
            """)
        # Main chat area
        with gr.Row():
            with gr.Column(scale=1):
                # Create initial welcome interface - fixed to messages format
                initial_welcome = [
                    {
                        "role": "assistant",
                        "content": """<div class="welcome-message">
    <div class="welcome-title">👋 Welcome to JarvisArt!</div>
    <div class="welcome-subtitle">Your AI-powered photo editing assistant is ready to help you transform your images with professional recommendations.</div>
    <div class="welcome-features">
        <div class="welcome-feature">
            <div class="welcome-feature-icon">📸</div>
            <div class="welcome-feature-title">Upload Image</div>
            <div class="welcome-feature-desc">Upload any photo you'd like to enhance</div>
        </div>
        <div class="welcome-feature">
            <div class="welcome-feature-icon">✨</div>
            <div class="welcome-feature-title">Describe Vision</div>
            <div class="welcome-feature-desc">Tell me your creative vision and style preferences</div>
        </div>
        <div class="welcome-feature">
            <div class="welcome-feature-icon">🎨</div>
            <div class="welcome-feature-title">Get Recommendations</div>
            <div class="welcome-feature-desc">Receive detailed editing suggestions tailored to your needs</div>
        </div>
    </div>
</div>"""
                    }
                ]
                
                chatbot = gr.Chatbot(
                    value=initial_welcome,
                    label="",
                    show_label=False,
                    elem_classes="chat-container",
                    height=600,
                    avatar_images=(USER_AVATAR_PATH, AI_AVATAR_PATH),
                    show_copy_button=True,
                    type='messages'  # Use new message format to avoid deprecation warnings
                )
        
        # Input area
        with gr.Row(elem_classes="input-section"):
            with gr.Column(scale=1):
                # Image upload area
                with gr.Group(elem_classes="image-upload-area"):
                    gr.Markdown("### 📸 Upload Your Image & Draw Bounding Box")
                    with gr.Column(elem_classes='jarvis_image_container'):
                        input_image = image_annotator(
                            label="Upload Image & Draw Bounding Box",
                            disable_edit_boxes=True,
                            image_type="filepath",
                            single_box=True,
                            show_label=False,
                            width=300
                        )
                    coordinates_output = gr.Textbox(label="BBox Coordinates", interactive=False)

                # Prompt input
                with gr.Group():
                    gr.Markdown("### 💬 Describe Your Vision")
                user_prompt = gr.Textbox(
                        label="",
                        show_label=False,
                        placeholder="Describe your desired editing style(Use '<box></box>' to represent the selected region of interest in the image.)... (e.g., 'I want a blue-toned look, a calm evening. Melancholy blue style')",
                    # value=default_user_prompt,
                    lines=3,
                        max_lines=5,
                        elem_classes="prompt-input"
                    )
                
                # Action buttons
                with gr.Row():
                    clear_btn = gr.Button(
                        "🗑️ Clear Chat", 
                        variant="secondary",
                        scale=1,
                        elem_classes="simple-button"
                    )
                    process_btn = gr.Button(
                        "✨ Generate Recommendations", 
                        variant="primary",
                        scale=1,
                        elem_classes="primary-simple-button"
                    )
                
                # System prompt viewing area (collapsible)
                with gr.Accordion("🔧 System Prompt Settings", open=False):
                    system_prompt_display = gr.Textbox(
                        value=system_prompt,
                        label="Current System Prompt (Preview)",
                        interactive=False,
                        lines=8,
                        max_lines=15
                    )
                    
                    gr.Markdown("""
                    **Note:** This system provides AI-powered image editing recommendations. 
                    Upload your image, describe your vision, and get professional editing suggestions powered by JarvisArt.
                    """)
                
                # Advanced parameter control panel (collapsible)
                with gr.Accordion("⚙️ Advanced Generation Parameters", open=False):
                    gr.Markdown("### Generation Parameter Controls")
                    
                    with gr.Row():
                        max_new_tokens = gr.Slider(
                            minimum=512,
                            maximum=20480,
                            value=10240,
                            step=256,
                            label="Max New Tokens",
                            info="Maximum number of tokens to generate"
                        )
                        
                    with gr.Row():
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=50,
                            step=1,
                            label="Top-K",
                            info="Sample from top K tokens with highest probability"
                        )
                        
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.8,
                            step=0.05,
                            label="Top-P (Nucleus Sampling)",
                            info="Cumulative probability threshold, controls generation diversity"
                        )
                    
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                            info="Generation randomness, higher values mean more random"
                        )
                    
                    with gr.Row():
                        reset_params_btn = gr.Button(
                            "🔄 Reset to Default",
                            variant="secondary",
                            size="sm"
                        )
                        
                        # Preset parameter buttons
                        conservative_btn = gr.Button(
                            "🎯 Conservative",
                            variant="secondary", 
                            size="sm"
                        )
                        
                        creative_btn = gr.Button(
                            "🎨 Creative",
                            variant="secondary",
                            size="sm"
                        )
                        
                        balanced_btn = gr.Button(
                            "⚖️ Balanced",
                            variant="secondary",
                            size="sm"
                        )
        
        # Quick examples
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 💡 Quick Examples")
                example_buttons = []
                examples = [
                    "Make it feel like a magical moment with vivid hues and a touch of fantasy style.",
                    "I want a blue-toned look, a calm evening. Melancholy blue style.",
                    "Create a vintage film look with soft highlights and muted tones.",
                    "Give it a Hong Kong cinema feel with vibrant contrasts and moody atmosphere.",
                    "Make this photo more vibrant and cinematic with warm colors."
                ]
                
                with gr.Row():
                    for i, example in enumerate(examples):
                        btn = gr.Button(
                            example[:40] + "...", 
                            size="sm",
                            scale=1,
                            elem_classes="example-buttons"
                        )
                        example_buttons.append(btn)
                        # Bind click event to set prompt
                        btn.click(
                            lambda ex=example: ex,
                            outputs=user_prompt
                        )
        
        # Event binding
        
        input_image.change(
            fn=get_box_coordinates,
            inputs=[input_image, user_prompt],
            outputs=[coordinates_output, user_prompt]
        )
        input_image.upload(
            fn=process_upload,
            inputs=[input_image],
            outputs=[input_image]
        )
        # Main processing button - streaming output, pass all parameters
        process_btn.click(
            fn=process_analysis_pipeline_stream,
            inputs=[input_image, user_prompt, max_new_tokens, top_k, top_p, temperature],
            outputs=[chatbot]
        )
        
        # Clear chat history
        clear_btn.click(
            lambda: [],
            outputs=[chatbot]
        )
        
        # Submit on Enter, pass all parameters
        user_prompt.submit(
            fn=process_analysis_pipeline_stream,
            inputs=[input_image, user_prompt, max_new_tokens, top_k, top_p, temperature],
            outputs=[chatbot]
        )
        
        # Parameter control events
        # Reset parameters
        reset_params_btn.click(
            lambda: (10240, 50, 0.8, 0.7),
            outputs=[max_new_tokens, top_k, top_p, temperature]
        )
        
        # Conservative mode: more deterministic output
        conservative_btn.click(
            lambda: (8192, 30, 0.6, 0.3),
            outputs=[max_new_tokens, top_k, top_p, temperature]
        )
        
        # Creative mode: more diverse output
        creative_btn.click(
            lambda: (12288, 80, 0.9, 1.0),
            outputs=[max_new_tokens, top_k, top_p, temperature]
        )
        
        # Balanced mode: balance determinism and creativity
        balanced_btn.click(
            lambda: (10240, 50, 0.8, 0.7),
            outputs=[max_new_tokens, top_k, top_p, temperature]
        )
    
    return demo

if __name__ == "__main__":
    local_dict={}
    print("="*60)
    print("🎨 Starting JarvisArt Image Editing Recommendation Assistant")
    print("="*60)
    print("🔧 System Configuration Information:")
    print(f"   🤖 Model Name: {args.model_name}")
    print(f"   📍 API Endpoint: {args.api_endpoint}:{args.api_port}")
    print(f"   🌐 Web Interface: {args.server_name}:{args.server_port}")
    # print(f"   🚀 CUDA Device: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
    print("="*60)

    # Connection check tips
    print("🔍 Connection Check Tips:")
    print("   If you encounter connection issues, please check:")
    print("   1. Is the API service running?")
    print("   2. Is the port occupied by another program?")
    print("   3. Do the firewall settings allow connections?")
    print("   4. If using SSH, ensure port forwarding is set up correctly")
    print()
    print("💡 Custom Parameter Examples:")
    print("   python demo_gradio.py --api_endpoint 192.168.1.100 --api_port 8001")
    print("="*60)
    
    demo = create_interface()
    
    # Launch the Gradio app on specified host and port
    print(f"🚀 Starting Web Interface...")
    try:
        demo.launch(
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share,  # Enable public sharing if requested
            show_error=True
        )
        print(f"✅ JarvisArt interface started successfully!")
        print(f"🌐 Local access: http://{args.server_name}:{args.server_port}")
        if args.share:
            print(f"🌍 Public sharing enabled! Check the 'public URL' link in the console output above")
            print(f"   Share link valid for: 72 hours")
            print(f"   ⚠️ Note: Anyone with the share link can access your app")
        else:
            print(f"ℹ️ Public sharing disabled, accessible only locally/within LAN")
            print(f"   Enable sharing: Add --share parameter")
        if args.server_name == "0.0.0.0":
            print(f"🌐 Local access: http://localhost:{args.server_port}")
            print(f"🌐 LAN access: http://<your IP address>:{args.server_port}")
    except Exception as e:
        print(f"❌ Gradio startup failed: {str(e)}")
        print("🔄 Trying other ports...")

        # Try other ports
        success = False
        for port in range(args.server_port + 1, args.server_port + 10):
            try:
                print(f"   Trying port {port}...")
                demo.launch(
                    server_name=args.server_name,
                    server_port=port,
                    share=False,
                    show_error=True
                )
                print(f"✅ JarvisArt interface started successfully!")
                print(f"🌐 Local access: http://{args.server_name}:{port}")
                if args.server_name == "0.0.0.0":
                    print(f"🌐 Local access: http://localhost:{port}")
                    print(f"🌐 LAN access: http://<your IP address>:{port}")
                success = True
                break
            except Exception as port_e:
                print(f"   Port {port} is also occupied")
                continue
        
        if not success:
            print("❌ No available ports found")
            print("💡 Solutions:")
            print("   1. Close other programs occupying the port")
            print("   2. Use --server_port parameter to specify a different port")
            print("   3. Check firewall settings")
            sys.exit(1)
