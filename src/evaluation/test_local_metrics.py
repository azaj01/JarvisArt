import os
import cv2
import numpy as np
from tqdm import tqdm
import json
import megfile
from PIL import Image
import csv
import argparse
from torch import nn
from torchvision import transforms
import copy


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

def calculate_metrics(mask_dir, test_json_path):
    crops = {
            "center": center_square_crop,
            "right": right_square_crop,
            "left": left_square_crop,
            "rightd": rightd_square_crop,
            "leftd": leftd_square_crop
    }
    with open(test_json_path, 'r', encoding='utf-8') as file:
        dataset_source = json.load(file)

    L1_sum = 0.0
    L2_sum = 0.0
    criterion_l1 = nn.L1Loss()
    criterion_l2 = nn.MSELoss()
    dict_list = []

    # 主循环
    for item_sample in tqdm(dataset_source, desc="Processing images"):
        temp_dict = {}
        src_path = item_sample["gt_path"]
        tar_path = item_sample["save_path"]

        temp_dict["edited_image"] = item_sample["save_path"]
        temp_dict["instruction"] = item_sample["instruction"]
        temp_dict["data_type"] = "global"
        if "portrait_final" in src_path:
            temp_dict["data_type"] = "local"

        if os.path.exists(src_path) == False or os.path.exists(tar_path) == False:
            print(f"Skipping {src_path} or {tar_path} as it does not exist.")
            continue


        mask_name = src_path.split("/")[-2] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)
        crop_func = crops[item_sample["crop_mode"]]

        src_img = cv2.cvtColor(np.array(crop_func(src_path)), cv2.COLOR_RGB2BGR) 
        tar_img = cv2.cvtColor(np.array(crop_func(tar_path)), cv2.COLOR_RGB2BGR) 
        if tar_img.shape != src_img.shape:
            tar_img = cv2.resize(tar_img, (src_img.shape[1], src_img.shape[0]), interpolation=cv2.INTER_AREA)

        if temp_dict["data_type"] == "global":
            temp_dict["l1_local"] = 0.0
            temp_dict["l2_local"] = 0.0
            dict_list.append(copy.deepcopy(temp_dict))
            continue

        mask = cv2.cvtColor(np.array(crop_func(mask_path)), cv2.COLOR_RGB2BGR)   

        if mask.shape != src_img.shape[:2]:
            mask = cv2.resize(mask, (src_img.shape[1], src_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask_3d = mask
        weights = np.ones_like(src_img, dtype=np.float32)
        weights[~mask_3d] = 0.5

        src_double = src_img.astype(np.float32) / 255.0
        tar_double = tar_img.astype(np.float32) / 255.0
        src_weighted = src_double * weights
        tar_weighted = tar_double * weights

        torch_image_gt_ = transforms.ToTensor()(src_weighted)
        torch_image_edit_ = transforms.ToTensor()(tar_weighted)
        l1 = criterion_l1(torch_image_gt_, torch_image_edit_).detach().cpu().numpy().item()
        l2 = criterion_l2(torch_image_gt_, torch_image_edit_).detach().cpu().numpy().item()

        L1_sum = L1_sum + l1
        L2_sum = L2_sum + l2
        temp_dict["l1_local"] = l1
        temp_dict["l2_local"] = l2

        dict_list.append(copy.deepcopy(temp_dict))

    return dict_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json_path", type=str, default="datasets/MMArt-Bench/datainfo.json")
    parser.add_argument("--mask_path", type=str, default="datasets/MMArt-Bench/local_mask")
    args = parser.parse_args()

    type_name_list = ["source"]

    for type_name in type_name_list:
        group_csv_list = calculate_metrics(args.mask_path, args.test_json_path.replace("<type_name>", type_name))
        
        # Calculate and print metrics only for local data_type
        l1_local_scores = []
        l2_local_scores = []
        
        for row in group_csv_list:
            if row.get('data_type') == 'local':
                if 'l1_local' in row and row['l1_local'] is not None:
                    l1_local_scores.append(float(row['l1_local']))
                if 'l2_local' in row and row['l2_local'] is not None:
                    l2_local_scores.append(float(row['l2_local']))
        
        # Print calculated metrics with title
        print(f"=== Local Metrics for {type_name} ===")
        if l1_local_scores:
            l1_mean = sum(l1_local_scores) / len(l1_local_scores)
            print(f"L1 Local Mean: {l1_mean:.4f}")
        
        if l2_local_scores:
            l2_mean = sum(l2_local_scores) / len(l2_local_scores)
            print(f"L2 Local Mean: {l2_mean:.4f}")
        print("=" * 40)