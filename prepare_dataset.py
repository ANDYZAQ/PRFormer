import os
import cv2
import numpy as np
from tqdm import tqdm
import torch


def make_dataset(split=0, data_root=None, data_list=None, sub_list=None, filter_intersection=False):    
    assert split in [0, 1, 2, 3]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2, 
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32    
    image_label_list = []  
    list_read = open(data_list).readlines()
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        new_label_class = []     

        if filter_intersection:  # filter images containing objects of novel categories during meta-training
            if set(label_class).issubset(set(sub_list)):
                for c in label_class:
                    if c in sub_list:
                        tmp_label = np.zeros_like(label)
                        target_pix = np.where(label == c)
                        tmp_label[target_pix[0],target_pix[1]] = 1 
                        if tmp_label.sum() >= 2 * 32 * 32:      
                            new_label_class.append(c)     
        else:
            for c in label_class:
                if c in sub_list:
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c)
                    tmp_label[target_pix[0],target_pix[1]] = 1 
                    if tmp_label.sum() >= 2 * 32 * 32:      
                        new_label_class.append(c)            

        label_class = new_label_class

        if len(label_class) > 0:
            image_label_list.append(item)
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list[c].append(item)
                    
    print("Checking image&label pair {} list done! ".format(split))
    return image_label_list, sub_class_file_list


def get_sub_lists(data_set, split):
    if data_set == 'pascal':
        class_list = list(range(1, 21))                         # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        if split == 3: 
            sub_list = list(range(1, 16))                       # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            sub_val_list = list(range(16, 21))                  # [16,17,18,19,20]
        elif split == 2:
            sub_list = list(range(1, 11)) + list(range(16, 21)) # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
            sub_val_list = list(range(11, 16))                  # [11,12,13,14,15]
        elif split == 1:
            sub_list = list(range(1, 6)) + list(range(11, 21))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
            sub_val_list = list(range(6, 11))                   # [6,7,8,9,10]
        elif split == 0:
            sub_list = list(range(6, 21))                       # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            sub_val_list = list(range(1, 6))                    # [1,2,3,4,5]

    elif data_set == 'coco':
        print('INFO: using SPLIT COCO (FWB)')
        class_list = list(range(1, 81))
        if split == 3:
            sub_val_list = list(range(4, 81, 4))
            sub_list = list(set(class_list) - set(sub_val_list))                    
        elif split == 2:
            sub_val_list = list(range(3, 80, 4))
            sub_list = list(set(class_list) - set(sub_val_list))    
        elif split == 1:
            sub_val_list = list(range(2, 79, 4))
            sub_list = list(set(class_list) - set(sub_val_list))    
        elif split == 0:
            sub_val_list = list(range(1, 78, 4))
            sub_list = list(set(class_list) - set(sub_val_list))    
        
    return sub_list, sub_val_list


if __name__=="__main__":
    dataname, split, tv = 'coco', 3, 'train'
    sub_list, sub_val_list = get_sub_lists(dataname, split)
    if dataname == 'coco':
        split_data_list="list/coco/" + tv + "_list_split" + str(split) + ".pth"
        data_root = "data/COCO2017"
        save_data_list = "list/coco_new/" + tv + "_list_split" + str(split) + ".pth"
    elif dataname in ['pascal', 'p2o']:
        split_data_list="list/pascal/voc_" + tv+ "_list_split" + str(split) + ".pth"

    if os.path.isfile(split_data_list):
        image_label_list, sub_class_file_list = torch.load(split_data_list)

    # print(image_label_list)
        
    image_label_list_new = []
    sub_class_file_list_new = {}
    for sub_c in sub_list:
        sub_class_file_list_new[sub_c] = []
    for imgpath, labelpath in tqdm(image_label_list):
        label_name = os.path.join(data_root, labelpath)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        new_label_class = []
        if set(label_class).issubset(set(sub_list)):
            for c in label_class:
                if c in sub_list:
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c)
                    tmp_label[target_pix[0],target_pix[1]] = 1 
                    if tmp_label.sum() >= 2 * 32 * 32:      
                        new_label_class.append(c)     
        
        label_class = new_label_class

        if len(label_class) > 0:
            image_label_list_new.append((imgpath, labelpath))
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list_new[c].append((imgpath, labelpath))

    print(len(image_label_list), len(image_label_list_new))

    torch.save((image_label_list_new, sub_class_file_list_new), save_data_list)