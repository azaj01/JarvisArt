import os
import json
from utils.xmp_lua_converter import parse_xmp_to_json, parse_lua_to_json
from system_prompt import LONG_SYSTEM_PROMPT, SHORT_SYSTEM_PROMPT,SHORT_SYSTEM_PROMPT_MULTILINGUAL
import tqdm
import concurrent.futures
import threading
import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Data processing script')
    parser.add_argument('--workers', type=int, default=100, help='Number of parallel worker threads')
    parser.add_argument('--tasks', nargs='+', default=["global_cot"], 
                        choices=["global_cot", "local_cot"], help='Types of tasks to process')
    parser.add_argument('--before_image', type=str, default="before.jpg", help='Original image filename')
    parser.add_argument('--roc_file', type=str, default="config.lua", help='Configuration filename')
    parser.add_argument('--user_intent_file', type=str, default="user_want.txt", help='User intent filename')
    parser.add_argument('--cot_file', type=str, default="revised160_expert_cot.txt", help='COT filename')
    parser.add_argument('--global_data_path', type=str, required=True, help='Path to global dataset')
    parser.add_argument('--local_data_path', type=str, help='Path to local dataset (required if processing local_cot)')
    parser.add_argument('--output_dir', type=str, default="./output", help='Output directory for JSON files')
    return parser.parse_args()

def get_task_paths(args):
    """Generate task paths based on arguments"""
    return {
        "global_cot": {
            "data_path": args.global_data_path,
            "save_path": os.path.join(args.output_dir, 'MMArt_global.json')
        },
        "local_cot": {
            "data_path": args.local_data_path,
            "save_path": os.path.join(args.output_dir, 'MMArt_local.json')
        }
    }

# Thread lock for safely appending data
lock = threading.Lock()
# Error counters
error_counters = {
    "config_parse_error": 0,
    "missing_image_file": 0,
    "missing_config_file": 0,
    "missing_user_intent_file": 0,
    "empty_user_intent": 0,
    "missing_cot_file": 0,
    "empty_cot_file": 0,
    "other_error": 0,
    "total_error": 0,
    "success": 0
}
error_counter_lock = threading.Lock()

def process_single_item(root, files, dataset_folder, file_names):
    """Process a single data item"""
    try:
        # Check if necessary files exist
        if file_names['before_image'] not in files:
            with error_counter_lock:
                error_counters["missing_image_file"] += 1
                error_counters["total_error"] += 1
            raise ValueError(f"Original image file not found: {os.path.join(root, file_names['before_image'])}")
            
        # Check roc file
        roc_file = None
        config = None
        if file_names['roc_file'] in files:
            roc_file = os.path.join(root, file_names['roc_file'])
        else:
            # If config.xmp doesn't exist, try to find config.lua
            lua_file = file_names['roc_file'].replace('.xmp', '.lua')
            if lua_file in files:
                roc_file = os.path.join(root, lua_file)
            else:
                with error_counter_lock:
                    error_counters["missing_config_file"] += 1
                    error_counters["total_error"] += 1
                raise ValueError(f"Configuration file not found (config.xmp or config.lua): {root}")
        
        # Process configuration based on file type
        if roc_file.endswith('.xmp'):
            config = parse_xmp_to_json(roc_file)
        elif roc_file.endswith('.lua'):
            # Use parse_lua_to_json to process lua file
            config = parse_lua_to_json(roc_file)
        
        if config is None:
            with error_counter_lock:
                error_counters["config_parse_error"] += 1
                error_counters["total_error"] += 1
            raise ValueError(f"Configuration file parsing error: {roc_file}")
            
        myimg_path = os.path.join(root, file_names['before_image'])
            
        # Check user intent file
        user_intend_path = os.path.join(root, file_names['user_intent_file'])
        if not os.path.exists(user_intend_path):
            with error_counter_lock:
                error_counters["missing_user_intent_file"] += 1
                error_counters["total_error"] += 1
            raise ValueError(f"User intent file not found: {user_intend_path}")
            
        with open(user_intend_path, 'r') as f:
            user_intend = f.read()
            
        if not user_intend:
            with error_counter_lock:
                error_counters["empty_user_intent"] += 1
                error_counters["total_error"] += 1
            raise ValueError(f"User intent file is empty: {user_intend_path}")
            
        # Try to get cot path
        cot_path = os.path.join(root, file_names['cot_file'])
        if not os.path.exists(cot_path):
            with error_counter_lock:
                error_counters["missing_cot_file"] += 1
                error_counters["total_error"] += 1
            raise ValueError(f"COT file not found: {cot_path}")
        
        with open(cot_path, 'r') as f:
            cot = f.read()
        
        if not cot:
            with error_counter_lock:
                error_counters["empty_cot_file"] += 1
                error_counters["total_error"] += 1
            raise ValueError(f"COT file is empty: {cot_path}")
            
        answer = f"<think>{cot}</think>\n<answer>{config}</answer>"
        
        # Create data entry
        data_entry = {
            "messages": [
                {
                    "content": f"<image>{user_intend}",
                    "role": "user"
                },
                {
                    "content": f"{answer}",
                    "role": "assistant"
                }
            ],
            "images": [f"{myimg_path}"],
            "system": SHORT_SYSTEM_PROMPT_MULTILINGUAL,
        }
        
        with error_counter_lock:
            error_counters["success"] += 1
            
        return data_entry
    except Exception as e:
        # If not a known error type, count as other error
        if not any(err_type in str(e) for err_type in ["Original image file not found", "Configuration file not found", 
                                                     "Configuration file parsing error", "User intent file not found", 
                                                     "User intent file is empty", "COT file not found", "COT file is empty"]):
            with error_counter_lock:
                error_counters["other_error"] += 1
                error_counters["total_error"] += 1
        print(f"Error: {e}, Path: {root}")
        return None

def process_directory(dataset_path, dataset_folder, num_workers, file_names):
    """Process a directory, return all valid data items"""
    results = []
    
    items_to_process = []
    # Collect all items to process
    for root, dirs, files in os.walk(dataset_path):
        if root != dataset_path:  # Skip the root directory
            items_to_process.append((root, files))
    
    total_items = len(items_to_process)
    
    with tqdm.tqdm(total=total_items, desc=f'{dataset_folder} progress') as pbar:
        # Use thread pool to process data
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_item = {
                executor.submit(process_single_item, root, files, dataset_folder, file_names): (root, files)
                for root, files in items_to_process
            }
            
            for future in concurrent.futures.as_completed(future_to_item):
                data_entry = future.result()
                if data_entry:
                    results.append(data_entry)
                pbar.update(1)
    
    return results

def process_task(task_type, num_workers, file_names):
    """Process specified type of task"""
    all_data = []
    data_path = TASK_PATHS[task_type]["data_path"]
    save_path = TASK_PATHS[task_type]["save_path"]
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Traverse dataset folders
    for dataset_folder in os.listdir(data_path):
        dataset_path = os.path.join(data_path, dataset_folder)
        if not os.path.isdir(dataset_path):
            continue
            
        print(f"Processing dataset: {dataset_folder}")
        
        # Process current dataset directory
        dataset_results = process_directory(dataset_path, dataset_folder, num_workers, file_names)
        
        # Safely merge results using thread lock
        with lock:
            all_data.extend(dataset_results)

    # Save as JSON file
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    task_name = "Global editing" if task_type == "global_cot" else "Local editing"
    print(f"{task_name} data processing completed, total {len(all_data)} entries")
    return len(all_data)

def main():
    """Main function"""
    args = parse_args()
    
    # Generate task paths from arguments
    global TASK_PATHS
    TASK_PATHS = get_task_paths(args)
    
    # Validate task types and paths
    for task in args.tasks:
        if task not in TASK_PATHS:
            raise ValueError(f"Unsupported task type: {task}")
        if task == "local_cot" and not args.local_data_path:
            raise ValueError("--local_data_path is required when processing local_cot task")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # File name configuration
    file_names = {
        'before_image': args.before_image,
        'roc_file': args.roc_file,
        'user_intent_file': args.user_intent_file,
        'cot_file': args.cot_file
    }
    
    print(f"Using {args.workers} worker threads for processing")
    print(f"Output directory: {args.output_dir}")
    print(f"File configuration: Image={file_names['before_image']}, Config={file_names['roc_file']}, " 
          f"User Intent={file_names['user_intent_file']}, COT={file_names['cot_file']}")
    
    total_entries = 0
    # Process each task
    for task in args.tasks:
        print(f"Starting task processing: {task}")
        print(f"Data path: {TASK_PATHS[task]['data_path']}")
        count = process_task(task, args.workers, file_names)
        total_entries += count
        print(f"Task {task} processing completed")
    
    print(f"All tasks completed, total {total_entries} entries generated")
    print("\nDetailed error statistics:")
    print(f"Successfully processed: {error_counters['success']}")
    print(f"Total failures: {error_counters['total_error']}")
    print("\nError statistics by type:")
    print(f"- Missing original image file: {error_counters['missing_image_file']}")
    print(f"- Missing configuration file: {error_counters['missing_config_file']}")
    print(f"- Configuration file parsing error: {error_counters['config_parse_error']}")
    print(f"- Missing user intent file: {error_counters['missing_user_intent_file']}")
    print(f"- Empty user intent file: {error_counters['empty_user_intent']}")
    print(f"- Missing COT file: {error_counters['missing_cot_file']}")
    print(f"- Empty COT file: {error_counters['empty_cot_file']}")
    print(f"- Other errors: {error_counters['other_error']}")

if __name__ == "__main__":
    main()
