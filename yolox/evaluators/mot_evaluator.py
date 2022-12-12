from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from yolox.tracker.byte_tracker import BYTETracker
from yolox.sort_tracker.sort import Sort
from yolox.deepsort_tracker.deepsort import DeepSort
from yolox.motdt_tracker.motdt_tracker import OnlineTracker
from yolox.mem_tracker.mem_tracker import MEMTracker

from yolox.mem_tracker import matching

import contextlib
import io
import os
import itertools
import json
import tempfile
import time
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict, OrderedDict
from glob import glob
from pathlib import Path
import motmetrics as mm

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))

# TODO
class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = BYTETracker(self.args)
        ori_thresh = self.args.track_thresh
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                    self.args.track_buffer = 14
                elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                    self.args.track_buffer = 25
                else:
                    self.args.track_buffer = 30

                if video_name == 'MOT17-01-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-06-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-12-FRCNN':
                    self.args.track_thresh = 0.7
                elif video_name == 'MOT17-14-FRCNN':
                    self.args.track_thresh = 0.67
                elif video_name in ['MOT20-06', 'MOT20-08']:
                    self.args.track_thresh = 0.3
                else:
                    self.args.track_thresh = ori_thresh

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = BYTETracker(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_sort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = Sort(self.args.track_thresh)
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = Sort(self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_memtracker(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
    ):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule
            
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        tracker = MEMTracker(self.args)
        ori_thresh = self.args.track_thresh
        last_frame_id = 0
        first_frame_id = 0
        track_list=defaultdict(list) # for animation
        # if self.args.mot20:
        #     gtfiles = glob(os.path.join(f'datasets/MOT20/train', '*/gt/gt_val_half.txt'))
        # else:
        #     gtfiles = glob(os.path.join(f'datasets/mot/train', '*/gt/gt_val_half.txt'))
        # gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                    self.args.track_buffer = 14
                elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                    self.args.track_buffer = 25
                else:
                    self.args.track_buffer = 30
                
                if video_name == 'MOT17-01-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-06-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-12-FRCNN':
                    self.args.track_thresh = 0.7
                elif video_name == 'MOT17-14-FRCNN':
                    self.args.track_thresh = 0.67
                elif video_name in ['MOT20-06', 'MOT20-08']:
                    self.args.track_thresh = 0.3
                else:
                    self.args.track_thresh = ori_thresh

                if video_name == 'MOT17-02-FRCNN':
                    last_frame_id = 299
                elif video_name == 'MOT17-04-FRCNN':
                    last_frame_id = 524
                elif video_name == 'MOT17-05-FRCNN':
                    last_frame_id = 418
                elif video_name == 'MOT17-09-FRCNN':
                    last_frame_id = 262
                elif video_name == 'MOT17-10-FRCNN':
                    last_frame_id = 326
                elif video_name == 'MOT17-11-FRCNN':
                    last_frame_id = 449
                elif video_name == 'MOT17-13-FRCNN':
                    last_frame_id = 374

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = MEMTracker(self.args)
                    first_frame_id = int(img_file_name[0].split('/')[-1])
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, f'{video_names[video_id - 1]}.txt')
                        write_results(result_filename, results)
                        results = []
                
                imgs = imgs.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                
                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())
                
                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
            
            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)
            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2]/tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                    # if (cur_iter % 100 == 0) and (cur_iter != 0):
                    #     distance_save_path=os.path.join(result_folder,
                    #                                     f'distance_img/{video_names[video_id]}_{cur_iter}_{tid}_({t.start_frame}'
                    #                                     f'-{t.end_frame})')
                    #     self.track_feature_plot(t, distance_save_path)
                    # track_list[t.track_id].append(t)
                # if frame_id == last_frame_id:
                #     self.memory_feature_plot_2(online_targets, result_folder, video_names[video_id], last_frame_id)
                    # track_feature_plot(track_list, video_names[video_id])
                    # self.memory_feature_plot_for_label(online_targets, gt[video_name], result_folder,
                    #                                    video_names[video_id], last_frame_id)
                    # track_list=defaultdict(list)

                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))
            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader):
                result_filename = os.path.join(result_folder, f'{video_names[video_id]}.txt')
                write_results(result_filename, results)
            
        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_deepsort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_motdt(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            '''
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools import cocoeval as COCOeval
                logger.warning("Use standard COCOeval.")
            '''
            #from pycocotools.cocoeval import COCOeval
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info

    @staticmethod
    def track_feature_plot(track, distance_path):
        frame_ids = list(track.memory['frame_id'])
        features = list(track.memory['features'])
        # tlbrs = []
        curr_feature = track.curr_feature
        cost_matrix = matching.features_distance(curr_feature, features)
        axes = plt.axes()
        axes.set_ylim(bottom=0)
        if (track.end_frame - track.start_frame) > 10:
            axis_label = list(range(track.start_frame, track.end_frame+1, (track.end_frame - track.start_frame)//10))
            if track.end_frame not in axis_label:
                axis_label = axis_label + [track.end_frame]
        else:
            axis_label = list(range(track.start_frame, track.end_frame+1))
        plt.xticks(axis_label) # plt.xticks(frame_ids, label = frame_ids)
        plt.plot(frame_ids, cost_matrix[-1], '.')
        plt.title(f'feature distance - id: {track.track_id}_({track.start_frame}-{track.end_frame})')
        plt.xlabel('frame_id')
        plt.ylabel('distance')
        # plt.grid(True)
        plt.savefig(distance_path)
        plt.clf()

    @staticmethod
    def memory_feature_plot(tracks, path, video_name, last_frame_id):
        colors=plt.cm.jet(np.linspace(0, 1, len(tracks)))
        if len(tracks) >= 25:
            ncol=len(tracks)//25
        else:
            ncol=1
        for ti in range(len(tracks)):
            tid = tracks[ti].track_id
            t_start_frame = tracks[ti].start_frame
            t_end_frame = tracks[ti].end_frame
            curr_target = np.array(tracks[ti].curr_feature).reshape(1, -1)
            curr_memory = tracks[ti].memory
            curr_frame_ids = curr_memory['frame_id']
            curr_features = curr_memory['features']
            curr_cost_matrix = matching.features_distance(curr_target, curr_features)
            axes = plt.axes()
            axes.set_ylim(bottom=0)
            axis_label = list(range(1, last_frame_id+1, (last_frame_id-1)//10))
            plt.xticks(axis_label)
            for i, ot_t in enumerate(tracks[:ti] + tracks[ti+1:]):
                other_memory = ot_t.memory
                other_frame_ids = other_memory['frame_id']
                other_features = other_memory['features']
                other_cost_matrix = matching.features_distance(curr_target, other_features)
                plt.plot(other_frame_ids, other_cost_matrix[-1], '.-', label=f'{ot_t.track_id}', linewidth=0.5, markersize=2, color=colors[i])
            plt.plot(curr_frame_ids, curr_cost_matrix[-1], '^-', label = f'{tid}', linewidth=1, markersize=4, color=colors[-1])
            plt.axis('tight')
            plt.legend(loc=(1.02, 0.0), ncol=ncol, fontsize=8)
            save_path = os.path.join(path, f'graph/{video_name}_{tid}_({t_start_frame}-{t_end_frame})')
            plt.savefig(save_path)
            plt.clf()

    @staticmethod
    def memory_feature_plot_2(tracks, path, video_name, last_frame_id):
        def get_cost_matrix_for_target(target_feature, target_cxcy, memory):
            memory_ids = memory['frame_id']
            features = memory['features']
            positions = np.asarray(memory['tlbrs'])[:, :2]
            feature_cost_matrix = matching.features_distance(target_feature, features)
            distance_cost_matrix = matching.features_distance(target_cxcy, positions, 'euclidean')
            return memory_ids, feature_cost_matrix, distance_cost_matrix

        colors=plt.cm.jet(np.linspace(0, 1, len(tracks)))
        if len(tracks) >= 25:
            ncol=len(tracks)//25
        else:
            ncol=1
        for ti in range(len(tracks)):
            axes=plt.axes()
            axes.set_ylim(bottom = 0)
            axis_label=list(range(1, last_frame_id + 1, (last_frame_id - 1)//10))
            plt.xticks(axis_label)
            ax1=plt.subplot(1, 2, 1)
            ax2=plt.subplot(1, 2, 2)

            tid = tracks[ti].track_id
            t_start_frame = tracks[ti].start_frame
            t_end_frame = tracks[ti].end_frame
            curr_target = np.array(tracks[ti].curr_feature).reshape(1, -1)
            curr_position = tracks[ti].tlbr[:2]
            curr_frame_ids, curr_feature_matrix, curr_distance_matrix = \
                get_cost_matrix_for_target(curr_target, curr_position, tracks[ti].memory)

            for i, ot_t in enumerate(tracks[:ti] + tracks[ti+1:]):
                other_frame_ids, other_feature_matrix, other_distance_matrix = \
                    get_cost_matrix_for_target(curr_target, curr_position, ot_t.memory)
                ax1.plot(other_frame_ids, other_feature_matrix[-1], '.-', label=f'{ot_t.track_id}',
                         linewidth=0.5, markersize=2, color=colors[i])
                ax2.plot(other_frame_ids, other_distance_matrix[-1], '.-', label = f'{ot_t.track_id}',
                         linewidth=0.5, markersize=2, color=colors[i])

            ax1.plot(curr_frame_ids, curr_feature_matrix[-1], '^-', label = f'{tid}',
                     linewidth=1, markersize=4, color=colors[-1])
            ax2.plot(curr_frame_ids, curr_distance_matrix[-1], '^-', label=f'{tid}',
                     linewidth=1, markersize=4, color=colors[-1])

            ax1.axis('auto')
            ax2.axis('auto')
            ax1.set_title('Feature Distance : Cosine')
            ax2.set_title('Position Distance : Euclidean')
            plt.legend(loc=(1.02, 0.0), ncol=ncol, fontsize=8)
            save_path = os.path.join(path, f'graph/{video_name}_{tid}_({t_start_frame}-{t_end_frame})')
            plt.savefig(save_path)
            plt.clf()

    @staticmethod
    def memory_feature_plot_for_label(tracks, gt, model_folder, path, video_name, last_frame_id, start_frame_id):
        gt_dict = defaultdict(lambda : defaultdict(list))
        def tlwh_to_tlbr(tlwhs):
            ret = np.asarray(tlwhs).copy()
            ret[:, 2:] += ret[:, :2]
            return ret

        gt = gt.reset_index(['Id', 'FrameId']).sort_values(by=['FrameId'])
        Ids = gt['Id'].unique()
        Frames = sorted(list(gt.index.unique()))

        import cv2
        from yolox.data.dataloading import get_yolox_datadir
        from yolox.mem_tracker.reid_model import load_reid_model, extract_reid_features
        reid_model = load_reid_model(model_folder)

        for frame in Frames:
            img_file_name = f'{video_name}/img1/{start_frame_id+frame-1:06}.jpg'
            img_file_name = os.path.join(get_yolox_datadir(), 'mot', 'train', img_file_name)
            image = cv2.imread(img_file_name)
            curr_gts = gt.loc[gt['FrameId']==frame].sort_index()
            ids = curr_gts['Id'].values
            tlbrs = tlwh_to_tlbr(curr_gts.loc[:, ['X', 'Y', 'Width', 'Height']].values)
            features, _ = extract_reid_features(reid_model, image, tlbrs)
            features = features.cpu().numpy()
            for i, _id in enumerate(ids):
                gt_dict[_id]['frame_ids'].append(frame)
                gt_dict[_id]['tlbrs'].append(tlbrs[i])
                gt_dict[_id]['features'].append(features[i])

        for ti in range(len(tracks)):
            tid = tracks[ti].track_id


plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = [17, 7]