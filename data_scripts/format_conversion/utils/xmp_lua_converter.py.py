from utils.xmp_converter import parse_xmp
from utils.lua_converter import LuaConverter
import copy

def is_paint_gesture(gestures):
    for gesture in gestures:
        if gesture['What'] == 'Mask/Paint':
            return True
    return False

def is_straight_line(value):
    if len(value)%2 != 0:
        return False
    for i in range(0, len(value), 2):
        if value[i+1] != value[i+2]:
            return False
    return True

def change_obj(json_obj):
    keys_to_delete = []
    for key, value in json_obj.items():
        if value == "0" or value == 0:
            keys_to_delete.append(key)
        if key in ["CameraProfile", "CameraProfileDigest", "GrainSeed"]:
            keys_to_delete.append(key)
        if key in ["ToneCurvePV2012", "ToneCurvePV2012Red", "ToneCurvePV2012Green", "ToneCurvePV2012Blue"]:
            if is_straight_line(value):
                keys_to_delete.append(key)
    
    if "Look" in json_obj:
        look = json_obj["Look"]
        if "Copyright" in look:
            del look["Copyright"]
        if "Parameters" in look:
            parameters = look["Parameters"]
            parameters_keys_to_delete = []
            for key in parameters.keys():
                if key in ["ToneCurvePV2012", "ToneCurvePV2012Red", "ToneCurvePV2012Green", "ToneCurvePV2012Blue"]:
                    if is_straight_line(parameters[key]):
                        parameters_keys_to_delete.append(key)
            for key in parameters_keys_to_delete:
                del parameters[key]
            look["Parameters"] = parameters
        json_obj["Look"] = look


    for key in keys_to_delete:
        del json_obj[key]
    masks = json_obj.get("MaskGroupBasedCorrections", None)
    if masks is None:
        return json_obj
    masks_copy = masks.copy()
    pop_masks_id = []
    masks_final = []
    for k, mask in enumerate(masks_copy):
        keys_to_delete = []
        for key, value in mask.items():
            if value == "0" or value == 0:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del mask[key]
        correction_masks = mask.get("CorrectionMasks", None)
        correction_masks_copy = correction_masks.copy()
        correction_masks_final = []
        pop_correction_id = []
        for i in range(len(correction_masks_copy) - 1, -1, -1):
            correction_mask = correction_masks_copy[i]
            if "MaskSubType" in correction_mask:
                if correction_mask["MaskSubType"] == 0 and "Gesture" in correction_mask and is_paint_gesture(correction_mask["Gesture"]):
                    # correction_masks.pop(i)
                    pop_correction_id.append(i)
            if correction_mask["What"] == "Mask/Aggregate":
                # correction_masks.pop(i)
                pop_correction_id.append(i)
        for i in range(len(correction_masks_copy)):
            if i not in pop_correction_id:
                correction_masks_final.append(copy.deepcopy(correction_masks_copy[i]))

        mask["CorrectionMasks"] = correction_masks_final
        if len(correction_masks_final) == 0:
            pop_masks_id.append(k)
            # masks.pop(k)
        
    for i in range(len(masks)):
        if i not in pop_masks_id:
            masks_final.append(copy.deepcopy(masks[i]))

    json_obj["MaskGroupBasedCorrections"] = masks_final
            
            
    return json_obj

def parse_xmp_to_lua(xmp_path):
    json_obj = parse_xmp_to_json(xmp_path)
    return LuaConverter.to_lua(json_obj)

def parse_xmp_to_json(xmp_path):
    with open(xmp_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if len(lines) > 5000:
            return None
    lua_table = parse_xmp(xmp_path)
    json_obj = LuaConverter.from_lua(lua_table.replace('return {', '{'))
    json_obj = change_obj(json_obj)
    return json_obj

def parse_lua_to_json(lua_path):
    with open(lua_path, "r", encoding="utf-8") as f:
        lua_table = f.read()
    json_obj = LuaConverter.from_lua(lua_table.replace('return {', '{'))
    json_obj = change_obj(json_obj)
    return json_obj
