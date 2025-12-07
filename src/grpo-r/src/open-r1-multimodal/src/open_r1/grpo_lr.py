# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

import cv2
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from transformers.utils import logging
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from open_r1.vlm_modules import *
from open_r1.utils.lrc_task_distribution import LightroomManager
from open_r1.lrc_tools_reward import roa_reward
from open_r1.utils.system_prompt import SHORT_SYSTEM_PROMPT_MULTILINGUAL
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from open_r1.qwen2_5vl_monkey_patch import (
    monkey_patch_qwen2_5vl_flash_attn, 
    monkey_patch_qwen2_5vl_forward, 
    monkey_patch_torch_load,
    monkey_patch_generation_config
)

# Apply monkey patches
monkey_patch_qwen2_5vl_flash_attn()
monkey_patch_torch_load()
monkey_patch_generation_config()

# Initialize logger and global variables
logger = logging.get_logger(__name__)
tokenizer = None
tool_manager = None  # Will be initialized with config in main()

# Configuration
SYSTEM_PROMPT = SHORT_SYSTEM_PROMPT_MULTILINGUAL
exclude_lightroom_keys = ["Copyright", "Name"]


def initialize_tokenizer(model_path: str):
    """Initialize the tokenizer from the model path."""
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def initialize_lightroom_manager(script_args: 'GRPOScriptArguments'):
    """
    Initialize the LightroomManager with configuration from script arguments.
    
    Args:
        script_args: Script arguments containing Lightroom configuration.
    
    Returns:
        Initialized LightroomManager instance.
    
    Raises:
        ValueError: If required Lightroom configuration is missing in training_args.yaml
    """
    global tool_manager
    if tool_manager is None:
        tool_manager = LightroomManager(
            server_host=script_args.lightroom_server_host,
            server_port=script_args.lightroom_server_port,
            results_dir=script_args.lightroom_results_dir,
            temp_dir=script_args.lightroom_temp_dir,
            max_retries=script_args.lightroom_max_retries,
            retry_delay=script_args.lightroom_retry_delay,
            backoff_factor=script_args.lightroom_backoff_factor,
            single_request_timeout=script_args.lightroom_single_request_timeout,
            total_timeout=script_args.lightroom_total_timeout,
            file_wait_timeout=script_args.lightroom_file_wait_timeout,
            upload_dir=script_args.lightroom_upload_dir,
            max_concurrent_tasks=script_args.lightroom_max_concurrent_tasks,
            debug=script_args.lightroom_debug,
            enable_fallback=script_args.lightroom_enable_fallback,
        )
    return tool_manager


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Script arguments for the GRPO training script."""
    
    # Data configuration
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    
    # Reward configuration
    reward_funcs: List[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={"help": "Choose reward method: 'default', 'mcp', ..."},
    )
    is_reward_customized_from_vlm_module: bool = field(
        default=False,
        metadata={"help": "Whether to use a customized reward from vlm module"},
    )
    
    # Image processing configuration
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    
    # Task configuration
    task_type: Optional[str] = field(
        default=None,
        metadata={"help": "Choose task type: 'default', 'gui', ..."},
    )
    
    # Lightroom configuration
    lightroom_server_host: str = field(
        default="127.0.0.1",
        metadata={"help": "Lightroom reverse server host address"},
    )
    lightroom_server_port: int = field(
        default=8081,
        metadata={"help": "Lightroom reverse server port"},
    )
    lightroom_upload_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Lightroom upload directory path"},
    )
    lightroom_results_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Lightroom results directory path"},
    )
    lightroom_temp_dir: Optional[str] = field(
        default="/tmp/lightroom_grpo",
        metadata={"help": "Lightroom temporary directory path"},
    )
    lightroom_max_retries: int = field(
        default=4,
        metadata={"help": "Maximum number of retries for Lightroom requests"},
    )
    lightroom_retry_delay: float = field(
        default=5.0,
        metadata={"help": "Initial retry delay in seconds"},
    )
    lightroom_backoff_factor: float = field(
        default=2.0,
        metadata={"help": "Exponential backoff factor for retries"},
    )
    lightroom_single_request_timeout: float = field(
        default=120.0,
        metadata={"help": "Single HTTP request timeout in seconds"},
    )
    lightroom_total_timeout: float = field(
        default=600.0,
        metadata={"help": "Total operation timeout including all retries in seconds"},
    )
    lightroom_file_wait_timeout: float = field(
        default=180.0,
        metadata={"help": "File wait timeout in seconds"},
    )
    lightroom_max_concurrent_tasks: int = field(
        default=4,
        metadata={"help": "Maximum number of concurrent Lightroom tasks"},
    )
    lightroom_debug: bool = field(
        default=False,
        metadata={"help": "Enable Lightroom debug mode"},
    )
    lightroom_enable_fallback: bool = field(
        default=True,
        metadata={"help": "Enable fallback mechanism for Lightroom"},
    )


@dataclass
class GRPOModelConfig(ModelConfig):
    """Model configuration for GRPO training."""
    freeze_vision_modules: bool = False



def calculate_eab(gt_img_path: str, pred_img_path: str) -> float:
    """
    Calculate normalized mean CIELAB color difference (Delta E_ab, CIE76) between two images.

    Args:
        gt_img_path: Path to the ground truth image.
        pred_img_path: Path to the predicted image.

    Returns:
        float: Normalized mean Delta E_ab value (range 0 to 1).
               0 indicates identical images in LAB space, 1 indicates large difference.
               Returns 0.0 if images cannot be loaded or are incompatible.
    """
    try:
        # Load images using OpenCV (BGR format by default)
        gt_img = cv2.imread(gt_img_path)
        pred_img = cv2.imread(pred_img_path)

        # Validate image loading
        if gt_img is None:
            logger.error(f"Failed to read ground truth image: {gt_img_path}")
            return 0.0
        if pred_img is None:
            logger.error(f"Failed to read predicted image: {pred_img_path}")
            return 0.0

        # Resize predicted image to match ground truth if shapes differ
        if gt_img.shape != pred_img.shape:
            logger.warning(
                f"Image dimensions differ ({gt_img.shape} vs {pred_img.shape}). "
                f"Resizing predicted image to match ground truth."
            )
            pred_img = cv2.resize(
                pred_img, 
                (gt_img.shape[1], gt_img.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        # Validate 3-channel color images
        if len(gt_img.shape) != 3 or gt_img.shape[2] != 3:
            logger.error(f"Ground truth image is not 3-channel: {gt_img.shape}")
            return 0.0
        if len(pred_img.shape) != 3 or pred_img.shape[2] != 3:
            logger.error(f"Predicted image is not 3-channel: {pred_img.shape}")
            return 0.0

        # Convert from BGR to CIELAB color space using float32 to avoid overflow
        gt_lab = cv2.cvtColor(gt_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        pred_lab = cv2.cvtColor(pred_img, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Calculate Delta E in LAB space
        delta_lab = gt_lab - pred_lab
        delta_e_sq = np.sum(delta_lab ** 2, axis=2)
        delta_e = np.sqrt(delta_e_sq)
        mean_delta_e = np.mean(delta_e)

        # Normalize using exponential decay (k=0.015)
        # Delta E=100 typically represents very large perceptual difference (e.g., black to white)
        k = 0.015
        score = np.exp(-k * mean_delta_e)
        normalized_score = np.clip(score, 0.0, 1.0)
        
        return normalized_score

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Error calculating Eab: {e}")
        return 0.0


def clean_text(text: str, exclude_chars: List[str] = ['\n', '\r']) -> str:
    """
    Extract and clean text content from answer tags.
    
    Args:
        text: Input text that may contain <answer> tags.
        exclude_chars: Characters to exclude (currently unused but kept for compatibility).
    
    Returns:
        Cleaned text content.
    """
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]
    return text


# ==================== Reward Functions ====================

def global_param_matching_reward(completions, solution, **kwargs):
    """Calculate reward based on global parameter matching using Lightroom."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        reward = roa_reward(
            clean_text(content), 
            clean_text(sol), 
            type="global_param_matching"
        )
        rewards.append(reward)
    
    # Debug logging if enabled
    if os.getenv("DEBUG_MODE") == "true":
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        log_path = os.getenv("LOG_PATH")
        problem = kwargs.get("problem")[0]
        
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} OUTPUT -------------\n")
            for content in contents:
                f.write(f"Content: {content}\n")
            f.write(f"Problem: {problem}\n")
    
    return rewards


def global_param_accuracy_reward(completions, solution, **kwargs):
    """Calculate reward based on global parameter accuracy using Lightroom."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        reward = roa_reward(
            clean_text(content), 
            clean_text(sol), 
            type="global_param_accuracy"
        )
        rewards.append(reward)
    
    return rewards


def local_param_accuracy_reward(completions, solution, **kwargs):
    """Calculate reward based on mask accuracy using Lightroom."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        reward = roa_reward(
            clean_text(content), 
            clean_text(sol), 
            type="mask_accuracy"
        )
        rewards.append(reward)
    
    return rewards

def perception_quality_reward(completions, solution, image_path, **kwargs):
    """
    Calculate reward based on Delta E_ab color difference between images.
    
    Processes images using Lightroom tool manager and compares the result
    with the ground truth using CIELAB color space.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol, img_path in zip(contents, solution, image_path):
        # Extract tool call content from completion
        tool_call_content = tool_manager.extract_tool_call_content(content)
        
        if tool_call_content is None:
            reward = 0.0
        else:
            # Process image using tool manager
            pred_img_path = tool_manager.process_image(img_path[0], tool_call_content)
            
            if pred_img_path is None:
                logger.warning("Predicted image path is None")
                reward = 0.0
            else:
                try:
                    reward = calculate_eab(img_path[0], pred_img_path)
                except Exception as e:
                    logger.error(f"Error in perception_quality_reward: {e}")
                    reward = 0.0
        
        rewards.append(reward)
    
    return rewards


def format_reward(completions, **kwargs):
    """
    Calculate reward based on completion format compliance.
    
    Checks if the completion follows the expected format:
    <think>...</think><answer>...</answer>
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]

    # Debug logging if enabled
    if os.getenv("DEBUG_MODE") == "true":
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        log_path = os.getenv("LOG_PATH")
        
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Format reward -------------\n")
            for content, match in zip(completion_contents, matches):
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {bool(match)}\n")

    return [1.0 if match else 0.0 for match in matches]

# Reward functions registry
reward_funcs_registry = {
    "format": format_reward,
    "global_param_matching": global_param_matching_reward,
    "global_param_accuracy": global_param_accuracy_reward,
    "local_param_accuracy": local_param_accuracy_reward,
    "perception_quality_reward": perception_quality_reward,
}


# ==================== Model Selection Functions ====================

def get_vlm_module(model_name_or_path: str):
    """
    Get the appropriate VLM module class based on model name.
    
    Args:
        model_name_or_path: Path or name of the model.
    
    Returns:
        VLM module class (Qwen2VLModule or InvernVLModule).
    
    Raises:
        ValueError: If model is not supported.
    """
    model_name_lower = model_name_or_path.lower()
    
    if "qwen" in model_name_lower:
        return Qwen2VLModule
    elif "internvl" in model_name_lower:
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")


# ==================== Data Processing Functions ====================

def load_and_process_data(script_args: GRPOScriptArguments) -> Dataset:
    """
    Load and process training data from JSON files.
    
    Args:
        script_args: Script arguments containing data file paths and reward methods.
    
    Returns:
        Processed Hugging Face Dataset.
    """
    data_files = script_args.data_file_paths.split(":")
    
    # Determine reward methods for each data file
    if script_args.reward_method is None:
        accu_reward_methods = ["default"] * len(data_files)
    else:
        accu_reward_methods = script_args.reward_method.split(":")
        assert len(accu_reward_methods) == len(data_files), (
            f"Number of reward methods must match number of data files: "
            f"{len(accu_reward_methods)} != {len(data_files)}"
        )
    
    all_data = []
    
    for data_file, accu_reward_method in zip(data_files, accu_reward_methods):
        logger.info(f"Loading data from: {data_file}")
        
        with open(data_file, 'r') as f:
            json_data = json.load(f)
            
            for item in json_data:
                # Store image paths if present
                if 'images' in item:
                    item['image_path'] = item['images']
                
                # Extract problem and solution from messages
                if 'messages' in item and len(item['messages']) >= 2:
                    user_message = item['messages'][0]
                    assistant_message = item['messages'][1]
                    
                    # Extract problem text
                    if user_message['role'] == 'user' and 'content' in user_message:
                        item['problem'] = user_message['content'].replace('<image>', '')
                    
                    # Extract solution, removing think tags if present
                    if assistant_message['role'] == 'assistant' and 'content' in assistant_message:
                        answer_content = assistant_message['content']
                        pattern = r"<think>.*?</think>\s*<answer>(.*?)</answer>"
                        answer_match = re.search(pattern, answer_content, re.DOTALL)
                        
                        if answer_match:
                            item['solution'] = answer_match.group(1).strip()
                        else:
                            # Fallback: use whole content and remove answer tags
                            item['solution'] = answer_content.strip().replace(
                                "<answer>", ""
                            ).replace("</answer>", "")
                
                # Clean up and set reward method
                del item['messages']
                item['accu_reward_method'] = item.get('accu_reward_method', accu_reward_method)
                all_data.append(item)
    
    logger.info(f"Loaded {len(all_data)} examples from {len(data_files)} files")
    return Dataset.from_list(all_data)


# ==================== Main Training Function ====================

def main(script_args: GRPOScriptArguments, training_args: GRPOConfig, model_args: GRPOModelConfig):
    """Main function for GRPO training."""
    
    # Initialize LightroomManager with configuration from script_args
    initialize_lightroom_manager(script_args)
    
    # Initialize VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    logger.info(f"Using VLM module: {vlm_module_cls.__name__}")

    # Setup reward functions
    if script_args.is_reward_customized_from_vlm_module:
        reward_funcs = [
            vlm_module_cls.select_reward_func(func, script_args.task_type) 
            for func in script_args.reward_funcs
        ]
    else:
        logger.info(f"Available reward functions: {list(reward_funcs_registry.keys())}")
        reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    logger.info(f"Using reward functions: {[f.__name__ for f in reward_funcs]}")

    # Load and process dataset
    dataset = load_and_process_data(script_args)

    def make_conversation_from_jsonl(example):
        """
        Convert dataset example to conversation format with system and user prompts.
        
        Args:
            example: Dataset example containing problem, solution, and optionally image paths.
        
        Returns:
            Dictionary with formatted conversation prompt.
        """
        base_result = {
            'problem': example['problem'],
            'solution': f"{example['solution']}",
            'accu_reward_method': example['accu_reward_method'],
        }
        
        # Build conversation with images if present
        if 'image_path' in example and example['image_path'] is not None:
            # Validate all image paths exist
            assert all(os.path.exists(p) for p in example['image_path']), (
                f"Image paths do not exist: {example['image_path']}"
            )
            
            base_result['image_path'] = [p for p in example['image_path']]
            base_result['prompt'] = [
                {
                    'role': 'system',
                    'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]
                },
                {
                    'role': 'user',
                    'content': [
                        *({'type': 'image', 'text': None} for _ in range(len(example['image_path']))),
                        {'type': 'text', 'text': example['problem']}
                    ]
                }
            ]
        else:
            # Text-only conversation
            base_result['prompt'] = [
                {
                    'role': 'system',
                    'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]
                },
                {
                    'role': 'user',
                    'content': [{'type': 'text', 'text': example['problem']}]
                }
            ]
        
        return base_result

    # Convert dataset to conversation format
    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)

    # Split dataset for validation if requested
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        logger.info(f"Splitting dataset with validation ratio: {script_args.val_split_ratio}")
        train_val_split = dataset.train_test_split(test_size=script_args.val_split_ratio)
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']
        logger.info(f"Train size: {len(splits['train'])}, Validation size: {len(splits['validation'])}")

    # Initialize tokenizer and trainer
    trainer_cls = VLMGRPOTrainer
    logger.info(f"Using trainer: {trainer_cls.__name__}")
    initialize_tokenizer(model_args.model_name_or_path)
    
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
    )

    # Resume from checkpoint if available
    checkpoint_exists = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoint_exists:
        logger.info(f"Resuming training from checkpoint: {checkpoint_exists[0]}")
        trainer.train(resume_from_checkpoint=True)
    else:
        logger.info("Starting training from scratch")
        trainer.train()

    # Save final model
    logger.info(f"Saving model to: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    
    if training_args.push_to_hub:
        logger.info("Pushing model to Hugging Face Hub")
        trainer.push_to_hub()



# ==================== Entry Point ====================

if __name__ == "__main__":
    # Parse command line arguments
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    # Apply DeepSpeed Zero3 monkey patch if needed
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        logger.info("DeepSpeed Zero3 detected, applying Qwen2.5-VL forward monkey patch")
        monkey_patch_qwen2_5vl_forward()
    
    # Start training
    main(script_args, training_args, model_args)
