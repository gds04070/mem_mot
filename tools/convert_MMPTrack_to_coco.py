import os
import json
import numpy as np
from PIL import Image

DATA_PATH = '/home/nvadmin/Dataset/MMPTracking/'
OUT_PATH = 'datasets/'
SPLITS = ['validation', 'train']
DEBUG = False

if __name__ == "__main__":
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        data_path = os.path.join(DATA_PATH, split)
        img_path = os.path.join(data_path, 'images')
        ann_path = os.path.join(data_path, 'labels')
        out_path = os.path.join(OUT_PATH, f'MMP_{split}.json')
        out = {'images': [], 'annotations': [], 'videos': [], 'categories': [{'id': 1, 'name': 'pedestrian'}]}
        ams = os.listdir(img_path)
        video_cnt=0
        for am in ams:
            am_img_path = os.path.join(img_path, am)
            am_ann_path = os.path.join(ann_path, am)
            seqs = os.listdir(am_img_path)
            image_cnt = 0
            ann_cnt = 0
            for seq in sorted(seqs):
                video_cnt += 1
                out['videos'].append({'id': video_cnt, 'file_name':seq})
                seq_path = os.path.join(am_img_path, seq)
                label_path = os.path.join(am_ann_path, seq)
                images = os.listdir(seq_path)

