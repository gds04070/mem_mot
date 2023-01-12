import os
import json
import numpy as np
from PIL import Image
import cv2

DATA_PATH = '/home/nvadmin/Dataset/MMPTracking/'
OUT_PATH = 'datasets/'
SPLITS = ['validation', 'train']
DEBUG = False

if __name__ == "__main__":
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        data_path = os.path.join(DATA_PATH, split) # /home/nvadmin/Dataset/MMPTracking/train
        img_path = os.path.join(data_path, 'images') # /home/nvadmin/Dataset/MMPTracking/train/images
        ann_path = os.path.join(data_path, 'labels') # /home/nvadmin/Dataset/MMPTracking/train/labels
        out_path = os.path.join(OUT_PATH, f'MMP_{split}.json')
        out = {'images': [], 'annotations': [], 'videos': [], 'categories': [{'id': 1, 'name': 'pedestrian'}]}
        ams = os.listdir(img_path) # 63am, 64am
        video_cnt=0
        image_cnt = 0
        for am in ams:
            am_img_path = os.path.join(img_path, am) # /home/nvadmin/Dataset/MMPTracking/train/images/63am
            am_ann_path = os.path.join(ann_path, am) # /home/nvadmin/Dataset/MMPTracking/train/labels/63am
            seqs = os.listdir(am_img_path) # cafe_shop_0, lobby_0, ...
            
            ann_cnt = 0
            for seq in sorted(seqs):
                video_cnt += 1
                out['videos'].append({'id': video_cnt, 'file_name':seq})
                seq_path = os.path.join(am_img_path, seq) # /home/nvadmin/Dataset/MMPTracking/train/images/63am/cafe_shop_0
                label_path = os.path.join(am_ann_path, seq) # /home/nvadmin/Dataset/MMPTracking/train/labels/63am/cafe_shop_0
                images = os.listdir(seq_path) # rgb_00000_1.jpg, rgb_00000_2.jpg, ...
                num_images = len([image for image in images if 'jpg' in image])
                for i in range(num_images):
                    img = cv2.imread(os.path.join(seq_path, images[i]))
                    height, width = img.shape[:2]
                    image_cnt += 1
                    _, frame_id, camera_id = list(map(int, images[i].split('.')[0].split('_')))
                    image_info = {
                        'file_name': f'{am}/{seq}/{images[i]}',
                        'id': image_cnt + i + 1, # image number in the entire training set 
                        'frame_id': frame_id,
                        'video_id': video_cnt,
                        'height': height,
                        'width':width,
                        'camera_id':camera_id,
                    }
                    out['images'].append(image_info)
                    label = json.load()

                

