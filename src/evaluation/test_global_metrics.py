from viescore import VIEScore
import json
import os
import megfile
from PIL import Image
from tqdm import tqdm
import argparse
import cv2
from torchvision.transforms import transforms
from torch import nn
import numpy as np
import time


def center_square_crop(image_path):
    with Image.open(image_path) as img:
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        right = left + s
        bottom = top + s
        img_cropped = img.crop((left, top, right, bottom))
        return img_cropped

def right_square_crop(image_path):
    with Image.open(image_path) as img:
        w, h = img.size
        s = min(w, h)
        left = 0
        top = 0
        right = left + s
        bottom = top + s
        img_cropped = img.crop((left, top, right, bottom))
        return img_cropped

def left_square_crop(image_path):
    with Image.open(image_path) as img:
        w, h = img.size
        s = min(w, h)
        left = w - s
        top = 0
        right = w 
        bottom = top + s
        img_cropped = img.crop((left, top, right, bottom))
        return img_cropped
        

def rightd_square_crop(image_path):
    with Image.open(image_path) as img:
        w, h = img.size
        s = min(w, h)
        left = 0
        top = h - s
        right = left + s
        bottom = top + s
        img_cropped = img.crop((left, top, right, bottom))
        return img_cropped

def leftd_square_crop(image_path):
    with Image.open(image_path) as img:
        w, h = img.size
        s = min(w, h)
        left = w - s
        top = h - s
        right = w 
        bottom = top + s
        img_cropped = img.crop((left, top, right, bottom))
        return img_cropped

def process_single_item(item, vie_score=None, max_retries=20):
    crops = {
                "center": center_square_crop,
                "right": right_square_crop,
                "left": left_square_crop,
                "rightd": rightd_square_crop,
                "leftd": leftd_square_crop
        }
    instruction = item['instruction']
    save_path_fullset_source_image = item['source_path']
    save_path_fullset_result_image = item['save_path']
    
    src_image_path = save_path_fullset_source_image
    save_path_item = save_path_fullset_result_image
    gt_image_path = item['gt_path']
    
    for retry in range(max_retries):
        try:
            if not os.path.exists(src_image_path) or not os.path.exists(save_path_item):
                print(f"File not found: {src_image_path} or {save_path_item}")
                return None
            crop_func = crops[item["crop_mode"]]
            pil_image_raw = Image.open(megfile.smart_open(src_image_path, 'rb')).convert("RGB")
            pil_image_edited = crop_func(megfile.smart_open(save_path_item, 'rb')).convert("RGB").resize((pil_image_raw.size[0], pil_image_raw.size[1]))
            text_prompt = instruction
            if "box_info" in item.keys():
                img_width, img_height = pil_image_raw.size
                if len(item['box_info']) > 0:
                    x1, y1, x2, y2 = item['box_info']
                    abs_x1 = x1 * img_width
                    abs_y1 = y1 * img_height
                    abs_x2 = x2 * img_width
                    abs_y2 = y2 * img_height
                    pil_image_raw_local = pil_image_raw.crop((abs_x1, abs_y1, abs_x2, abs_y2))
                    pil_image_edited_local = pil_image_edited.crop((abs_x1, abs_y1, abs_x2, abs_y2))
                    score_local_list = vie_score.evaluate([pil_image_raw_local, pil_image_edited_local], text_prompt, local_flag=True)
                    sc_local, pq_local, O_score_local = score_local_list
                    IsLocal = True
                else:
                    sc_local, pq_local, O_score_local = 0, 0, 0
                    IsLocal = False
            else:
                sc_local, pq_local, O_score_local = 0, 0, 0
                IsLocal = False
            
            score_list = vie_score.evaluate([pil_image_raw, pil_image_edited], text_prompt)
            sc_global, pq_global, O_score = score_list

            numpy_image_gt = cv2.imread(gt_image_path)
            numpy_image_edit = cv2.cvtColor(
                    np.array(crop_func(save_path_item)), cv2.COLOR_RGB2BGR 
                )
            height, width = numpy_image_gt.shape[:2] 
            numpy_image_edit = cv2.resize(numpy_image_edit, (width, height))

            criterion_l1 = nn.L1Loss()
            criterion_l2 = nn.MSELoss()
            numpy_image_gt_ = transforms.ToTensor()(numpy_image_gt)
            numpy_image_edit_ = transforms.ToTensor()(numpy_image_edit)
            l1 = criterion_l1(numpy_image_gt_, numpy_image_edit_).detach().cpu().numpy().item()
            l2 = criterion_l2(numpy_image_gt_, numpy_image_edit_).detach().cpu().numpy().item()

            return {
                "source_image": src_image_path,
                "edited_image": save_path_item,
                "instruction": instruction,
                "IsLocal": IsLocal,
                "data_type": save_path_item.split("/")[-2],
                "sc_global": sc_global,
                "sc_local": sc_local,
                "pq_global": pq_global,
                "pq_local": pq_local,
                "overall_score": O_score,
                "overall_score_local": O_score_local,
                "l1": l1,
                "l2": l2,
            }
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Error processing {save_path_item} (attempt {retry + 1}/{max_retries}): {e}")
                time.sleep(3)  # Optional: wait before retrying
                print(f"Retrying {save_path_item}...")
            else:
                print(f"Failed to process {save_path_item} after {max_retries} attempts: {e}")
                return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_json_path", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="google")#qwen25vl
    args = parser.parse_args()
    backbone = args.backbone
    
    with open(args.model_json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    vie_score = VIEScore(backbone=backbone, task="tie", key_path='secret_t2.env')

    futures = []
    group_list = []
    for item in tqdm(dataset):
        future = process_single_item(item, vie_score)
        futures.append(future)
    
    for future in tqdm(futures, total=len(futures)):
        if future:
            group_list.append(future)

    # Calculate metrics directly
    total_items = len(group_list)
    local_items = [item for item in group_list if item['IsLocal']]
    local_count = len(local_items)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Total items processed: {total_items}")
    print(f"Items with local evaluation: {local_count}")
    
    if total_items > 0:
        # Global metrics (for all items)
        avg_sc_global = sum(item['sc_global'] for item in group_list) / total_items
        avg_pq_global = sum(item['pq_global'] for item in group_list) / total_items
        avg_overall_global = sum(item['overall_score'] for item in group_list) / total_items
        avg_l1 = sum(item['l1'] for item in group_list) / total_items
        avg_l2 = sum(item['l2'] for item in group_list) / total_items
        
        print(f"\nGlobal Metrics (all {total_items} items):")
        print(f"  Average SC Global: {avg_sc_global:.4f}")
        print(f"  Average PQ Global: {avg_pq_global:.4f}")
        print(f"  Average Overall Score: {avg_overall_global:.4f}")
        print(f"  Average L1 Loss: {avg_l1:.4f}")
        print(f"  Average L2 Loss: {avg_l2:.4f}")
        
        # Local metrics (only for items where IsLocal=True)
        if local_count > 0:
            avg_sc_local = sum(item['sc_local'] for item in local_items) / local_count
            avg_pq_local = sum(item['pq_local'] for item in local_items) / local_count
            avg_overall_local = sum(item['overall_score_local'] for item in local_items) / local_count
            
            print(f"\nLocal Metrics (only {local_count} items with IsLocal=True):")
            print(f"  Average SC Local: {avg_sc_local:.4f}")
            print(f"  Average PQ Local: {avg_pq_local:.4f}")
            print(f"  Average Overall Score Local: {avg_overall_local:.4f}")
        else:
            print(f"\nNo items with local evaluation found.")