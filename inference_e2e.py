"""
JarvisArt End-to-End Inference Script
Single-round AI-powered image editing with Lightroom integration
"""

import os
import json
import base64
import re
import shutil
import threading
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tqdm
from openai import OpenAI

from utils.system_prompt import SHORT_SYSTEM_PROMPT_WITH_THINKING
from utils.lua_converter import LuaConverter
from utils.lrc_tools import LightroomManager
from tools.lua2lrt import lua_to_lrtemplate


# ============================================================================
# Response Wrapper
# ============================================================================

class Response:
    """Wrapper for API response text"""
    def __init__(self, text):
        self.response_text = text


# ============================================================================
# API Client
# ============================================================================

class APIClient:
    """OpenAI-compatible API client for vision-language models"""
    
    def __init__(self, api_endpoint, api_port, model_name="qwen2_vl", api_key="0", api_timeout=30):
        """
        Initialize API client
        
        Args:
            api_endpoint: API server address
            api_port: API server port
            model_name: Model identifier
            api_key: Authentication key
            api_timeout: API connection timeout in seconds
        """
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.api_port = api_port
        self.api_timeout = api_timeout
        self.api_connected = False
        
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url=f"http://{api_endpoint}:{api_port}/v1",
                timeout=api_timeout
            )
            self.api_connected = True
        except Exception as e:
            print(f"❌ API client initialization failed: {e}")
            print("⚠️ Program will continue but API functionality unavailable")
            self.client = None
            self.api_connected = False
    
    def chat(self, messages, system=None, images=None, default_timeout=180, **kwargs):
        """
        Send chat request with optional images
        
        Args:
            messages: List of conversation messages
            system: Optional system prompt
            images: Optional list of image paths
            default_timeout: Request timeout in seconds
            **kwargs: Additional API parameters
            
        Returns:
            List containing Response object
        """
        try:
            formatted_messages = self._format_messages(messages, system, images)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                stream=False,
                timeout=default_timeout,
                **kwargs
            )
            return [Response(response.choices[0].message.content)]
        except Exception as e:
            print(f"❌ API call error: {e}")
            return [Response(f"API call failed: {e}")]
    
    def _format_messages(self, messages, system, images):
        """Format messages with system prompt and images"""
        formatted = []
        
        if system:
            formatted.append({"role": "system", "content": system})
        
        image_idx = 0
        for msg in messages:
            if images and msg["role"] == "user" and image_idx < len(images):
                content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(images[image_idx])}"}
                    },
                    {"type": "text", "text": msg["content"]}
                ]
                formatted.append({"role": msg["role"], "content": content})
                image_idx += 1
            else:
                formatted.append(msg)
        
        return formatted


# ============================================================================
# Utility Functions
# ============================================================================

def encode_image(image_path):
    """Encode image file to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def compact_text(text):
    """Remove excessive whitespace and line breaks"""
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\n\s+', '\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text


def extract_tag_content(text, tag):
    """Extract content from XML-style tags"""
    pattern = f'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_json_from_answer(answer):
    """
    Extract configuration data from the answer string
    
    Args:
        answer: The answer string containing configuration data
        
    Returns:
        List with exactly one configuration object, or empty list
    """
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
    
    # Method 2: Fallback - find Python dict pattern
    dict_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(dict_pattern, answer, re.DOTALL)
    
    largest_match = None
    largest_size = 0
    
    for match in matches:
        try:
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


def save_lua_preset(json_data, output_path):
    """
    Convert JSON data to Lua preset file
    
    Args:
        json_data: Dict or JSON string with Lightroom parameters
        output_path: Path to save the Lua file
        
    Returns:
        Tuple of (file_path, error_message)
    """
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Parse JSON if it's a string
        if isinstance(json_data, str):
            try:
                json_obj = json.loads(json_data)
            except:
                return None, "Error parsing JSON: Invalid JSON format"
        else:
            json_obj = json_data
        
        # Convert to Lua format
        try:
            lua_content = LuaConverter.to_lua(json_obj)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write('return %s' % lua_content)
            return output_path, None
        except Exception as e:
            return None, f"Error writing Lua file: {str(e)}"
    
    except Exception as e:
        return None, f"Error in save_lua_preset: {str(e)}"


# ============================================================================
# Conversation History Management
# ============================================================================

class ConversationManager:
    """Manages conversation history storage and retrieval"""
    
    @staticmethod
    def save_single_round(image_path, user_prompt, full_response, 
                         tool_call, output_image=None, success=True):
        """
        Save single-round conversation data
        
        Args:
            image_path: Input image path
            user_prompt: User instruction
            full_response: Complete model response
            tool_call: Extracted tool call content
            output_image: Output image path
            success: Whether processing succeeded
            
        Returns:
            Dict with conversation data
        """
        thinking = extract_tag_content(full_response, 'think')
        answer = extract_tag_content(full_response, 'answer')
        
        return {
            "round": 1,
            "input_image": image_path,
            "user_prompt": user_prompt,
            "full_response": full_response,
            "thinking": thinking,
            "answer": answer,
            "tool_call": tool_call,
            "output_image": output_image,
            "success": success
        }
    
    @staticmethod
    def save_to_file(conversation_data, output_dir):
        """Save conversation history to JSON file"""
        try:
            history_file = os.path.join(output_dir, "conversation_history.json")
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved conversation history: {history_file}")
        except Exception as e:
            print(f"⚠️ Failed to save conversation history: {e}")


# ============================================================================
# Main Inference Pipeline
# ============================================================================

def run_single_round_inference(image_path, user_prompt, chat_model, 
                               lightroom_manager, save_base_path, 
                               system_prompt, default_timeout=180):
    """
    Execute single-round AI-powered image editing
    
    Args:
        image_path: Path to input image
        user_prompt: User editing instruction
        chat_model: API client instance
        lightroom_manager: Lightroom processing manager
        save_base_path: Base directory for results
        system_prompt: System prompt for model
        default_timeout: Request timeout in seconds
        
    Returns:
        None (saves results to disk)
    """
    try:
        # Setup result directories
        image_basename = Path(image_path).parent.name
        result_dir = os.path.join(save_base_path, image_basename)
        
        # Skip if already processed
        if os.path.exists(os.path.join(result_dir, "conversation_history.json")):
            print(f"⏭️ Already processed: {image_path}")
            return
        
        os.makedirs(result_dir, exist_ok=True)
        
        # Copy original image
        if os.path.exists(image_path):
            original_image = os.path.join(result_dir, os.path.basename(image_path))
            shutil.copy2(image_path, original_image)
        
        print(f"\n{'=' * 80}")
        print(f"PROCESSING: {image_basename}")
        print(f"{'=' * 80}")
        
        # Build messages
        messages = [{
            "role": "user",
            "content": user_prompt
        }]
        
        # Get model response
        responses = chat_model.chat(
            messages=messages,
            system=system_prompt,
            images=[image_path],
            default_timeout=default_timeout
        )
        
        full_response = responses[0].response_text
        
        if not full_response:
            print("⚠️ Empty response received")
            ConversationManager.save_to_file(
                ConversationManager.save_single_round(
                    image_path, user_prompt, "", None, None, False
                ),
                result_dir
            )
            return
        
        # Parse response
        answer = extract_tag_content(full_response, 'answer')
        tool_call = None
        
        if answer:
            # Extract JSON configuration from answer
            json_objects = extract_json_from_answer(answer)
            if json_objects:
                tool_call = json_objects[0]
        
        if not tool_call:
            print("⚠️ No valid tool call found in response")
            ConversationManager.save_to_file(
                ConversationManager.save_single_round(
                    image_path, user_prompt, full_response, None, None, False
                ),
                result_dir
            )
            return
        
        # Save Lua preset
        lua_path = os.path.join(result_dir, "output_image.lua")
        lua_file, error = save_lua_preset(tool_call, lua_path)
        
        if error:
            print(f"❌ Error saving Lua preset: {error}")
            ConversationManager.save_to_file(
                ConversationManager.save_single_round(
                    image_path, user_prompt, full_response, str(tool_call), None, False
                ),
                result_dir
            )
            return
        
        # Convert to lrtemplate format
        try:
            lrtemplate_path = lua_to_lrtemplate(lua_path)
            print(f"✅ Generated lrtemplate: {lrtemplate_path}")
        except Exception as e:
            print(f"⚠️ Failed to convert to lrtemplate: {e}")
        
        # Process image with Lightroom
        print("🔄 Processing image with Lightroom...")
        processed_image = lightroom_manager.process_image(image_path, str(tool_call))
        
        if processed_image and os.path.exists(processed_image):
            # Copy to result directory
            output_image = os.path.join(result_dir, "processed.jpg")
            shutil.copy2(processed_image, output_image)
            print(f"✅ Processed image saved: {output_image}")
        else:
            print("⚠️ Lightroom processing failed or returned no image")
            output_image = None
        
        # Save conversation history
        ConversationManager.save_to_file(
            ConversationManager.save_single_round(
                image_path, user_prompt, full_response, 
                str(tool_call), output_image, True
            ),
            result_dir
        )
        
        # Save text response
        response_file = os.path.join(result_dir, "response.txt")
        with open(response_file, "w", encoding="utf-8") as f:
            f.write(full_response)
        
        print(f"✅ Completed: {image_basename}")
    
    except Exception as e:
        print(f"❌ Inference error: {e}")
        # Save error state
        image_basename = Path(image_path).parent.name
        result_dir = os.path.join(save_base_path, image_basename)
        os.makedirs(result_dir, exist_ok=True)
        ConversationManager.save_to_file(
            ConversationManager.save_single_round(
                image_path, user_prompt, str(e), None, None, False
            ),
            result_dir
        )


# ============================================================================
# Batch Processing
# ============================================================================

def process_single_image(path, image_base_path, chat_model, lightroom_manager,
                        system_prompt, save_base_path, prompt_file_name,
                        default_timeout=180):
    """
    Process single image with AI editing
    
    Args:
        path: Relative path to image directory
        image_base_path: Base directory containing images
        chat_model: API client instance
        lightroom_manager: Lightroom manager
        system_prompt: System prompt
        save_base_path: Results directory
        prompt_file_name: User prompt filename
        default_timeout: Request timeout in seconds
        
    Returns:
        Status message
    """
    try:
        base_path = os.path.join(image_base_path, path)
        
        # Skip if already processed
        result_dir = os.path.join(save_base_path, path)
        if os.path.exists(os.path.join(result_dir, "conversation_history.json")):
            return f"⏭️ Skipped (already processed): {path}"
        
        # Find image file
        image_path = os.path.join(base_path, "before.jpg")
        if not os.path.exists(image_path):
            for temp_name in ["before.png", "input.jpg", "input.png"]:
                temp_path = os.path.join(base_path, temp_name)
                if os.path.exists(temp_path):
                    image_path = temp_path
                    break
        
        if not os.path.exists(image_path):
            return f"⚠️ Skipped (no image): {path}"
        
        # Read user prompt
        prompt_path = os.path.join(base_path, prompt_file_name)
        if not os.path.isfile(prompt_path):
            return f"⚠️ Skipped (no prompt): {path}"
        
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                user_prompt = f.read().strip()
        except Exception as e:
            return f"❌ Error reading prompt: {path} - {e}"
        
        # Process image
        thread_id = threading.current_thread().ident
        print(f"[Thread {thread_id}] Processing: {image_path}")
        
        run_single_round_inference(
            image_path, user_prompt, chat_model, lightroom_manager,
            save_base_path, system_prompt, default_timeout
        )
        
        print(f"[Thread {thread_id}] ✅ Completed: {path}")
        return f"✅ Completed: {path}"
    
    except Exception as e:
        thread_id = threading.current_thread().ident
        error_msg = f"[Thread {thread_id}] ❌ Error: {path} - {e}"
        print(error_msg)
        return error_msg


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="JarvisArt End-to-End Inference - Single-round AI image editing with Lightroom"
    )
    
    # API Configuration
    parser.add_argument("--api_endpoint", type=str, default="localhost",
                       help="API server address")
    parser.add_argument("--api_port", type=int, nargs='+', default=[8002],
                       help="API server port(s) for load balancing")
    parser.add_argument("--api_key", type=str, default="0",
                       help="API authentication key")
    parser.add_argument("--model_name", type=str, default="qwen2_vl",
                       help="AI model name")
    
    # Processing Configuration
    parser.add_argument("--max_threads", type=int, default=10,
                       help="Maximum concurrent threads")
    
    # File Paths
    parser.add_argument("--image_path", type=str, required=True,
                       help="Input image directory")
    parser.add_argument("--save_base_path", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--prompt_file_name", type=str, default="user_want.txt",
                       help="User prompt filename")
    
    # Processing Parameters
    parser.add_argument("--default_timeout", type=int, default=180,
                       help="Default timeout for API requests in seconds")
    parser.add_argument("--api_timeout", type=int, default=30,
                       help="API connection timeout in seconds")
    
    args = parser.parse_args()
    
    # Set default save path if not provided
    if args.save_base_path is None:
        args.save_base_path = os.path.join(args.image_path, "results")
    
    # Initialize API clients
    api_ports = args.api_port if isinstance(args.api_port, list) else [args.api_port]
    chat_models = [
        APIClient(args.api_endpoint, port, args.model_name, args.api_key, args.api_timeout)
        for port in api_ports
    ]
    
    # Initialize Lightroom manager
    lightroom_manager = LightroomManager()
    
    # System prompt
    system_prompt = SHORT_SYSTEM_PROMPT_WITH_THINKING
    
    # Get image list
    image_dirs = sorted([
        d for d in os.listdir(args.image_path)
        if os.path.isdir(os.path.join(args.image_path, d))
    ])
    
    print(f"Processing {len(image_dirs)} images with {args.max_threads} threads")
    print(f"Results will be saved to: {args.save_base_path}")
    
    # Process images concurrently
    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        futures = {
            executor.submit(
                process_single_image,
                path,
                args.image_path,
                chat_models[idx % len(chat_models)],  # Load balancing
                lightroom_manager,
                system_prompt,
                args.save_base_path,
                args.prompt_file_name,
                args.default_timeout
            ): path
            for idx, path in enumerate(image_dirs)
        }
        
        # Monitor progress
        with tqdm.tqdm(total=len(image_dirs), desc="Processing") as pbar:
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    print(f"\n{result}")
                except Exception as e:
                    print(f"\n❌ Exception for {path}: {e}")
                finally:
                    pbar.update(1)
    
    print(f"\n✅ Processing complete: {len(image_dirs)} images")


if __name__ == "__main__":
    main()
