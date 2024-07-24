# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation import SemSegEvaluator
from .util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU

class PropSemSegEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        post_process_func=None,
        dataname = 'pascal',
        split = None
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        meta = MetadataCatalog.get(dataset_name)
        self.ignore_label = ignore_label
        try:
            self._evaluation_set = meta.evaluation_set
        except AttributeError:
            self._evaluation_set = None
        self.post_process_func = (
            post_process_func
            if post_process_func is not None
            else lambda x, **kwargs: x
        )

        self.intersection_meter = AverageMeter()
        self.union_meter = AverageMeter()
        self.target_meter = AverageMeter()

        if dataname == 'coco':
            self.split_gap = 20
        elif dataname in ['pascal', 'p2o']:
            self.split_gap = 5
        elif dataname == 'c2pv':
            if split in [0,2]:
                self.split_gap = 6
            elif split in [1,3]:
                self.split_gap = 4
        self.class_intersection_meter = [0]*self.split_gap
        self.class_union_meter = [0]*self.split_gap  

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs of a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. 
                It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "'few_shot_result'".
        """

        in_img, in_label, out_label = [],[],[]
        mIoUs = []

        bs = len(inputs)
        for i in range(bs):

            target = inputs[i]['label'].unsqueeze(0).cuda(non_blocking=True)
            subcls = inputs[i]['subcls_list'][0]
            # output = outputs['few_shot_result'][i].unsqueeze(0).cuda(non_blocking=True)
            proposals = outputs['proposals'][i].unsqueeze(0).cuda(non_blocking=True)
            # subcls = torch.tensor(subcls)
            # print(subcls)

            # output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)         
            # output = output.max(1)[1]
            proposals = F.interpolate(proposals, size=target.size()[1:], mode='bilinear', align_corners=True).squeeze(0)
            target_remove_ignore = target.clone()
            target_remove_ignore[target == 255] = 0
            iop = (proposals * target_remove_ignore).sum(dim=(1,2)) / (proposals.sum(dim=(1,2)) + 1e-10)
            iop_args = torch.argsort(iop, descending=True)

            pseudo_proposal = proposals[iop_args[0]]
            iou = (pseudo_proposal * target_remove_ignore).sum() / ((pseudo_proposal + target_remove_ignore).sum() + 1e-10)
            selected_proposal = pseudo_proposal
            for iop_arg in iop_args[1:]:
                next_pseudo_proposal = torch.max(pseudo_proposal, proposals[iop_arg])
                # if (next_pseudo_proposal * target_remove_ignore).sum() / (next_pseudo_proposal + target_remove_ignore).sum() < iou - 1e-2:
                #     break
                iou_next = torch.max(iou, (next_pseudo_proposal * target_remove_ignore).sum() / ((next_pseudo_proposal + target_remove_ignore).sum() + 1e-10))
                if iou_next > iou:
                    iou = iou_next
                    selected_proposal = next_pseudo_proposal
                pseudo_proposal = next_pseudo_proposal
            
            output = torch.stack([1 - selected_proposal, selected_proposal], dim=0) # 2 x H x W
            output = output.max(0)[1].unsqueeze(0)

            intersection, union, new_target = intersectionAndUnionGPU(output, target, 2, 255)  # f-b iou, 2:fg and bg, 255: ignore label
            intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), new_target.cpu().numpy()
            self.intersection_meter.update(intersection), self.union_meter.update(union), self.target_meter.update(new_target)

            self.class_intersection_meter[(subcls-1)%self.split_gap] += intersection[1]
            self.class_union_meter[(subcls-1)%self.split_gap] += union[1] 
            mIoUs.append(intersection[1] / (union[1] + 1e-10))
        return mIoUs


    def evaluate(self):
        """
        Evaluates few shot semantic segmentation metrics (https://github.com/YanFangCS/CyCTR-Pytorch):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fbIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        """

        iou_class = self.intersection_meter.sum / (self.union_meter.sum + 1e-10)
        accuracy_class = self.intersection_meter.sum / (self.target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(self.intersection_meter.sum) / (sum(self.target_meter.sum) + 1e-10)

        
        class_iou_class = []
        class_miou = 0
        for i in range(len(self.class_intersection_meter)):
            class_iou = self.class_intersection_meter[i]/(self.class_union_meter[i]+ 1e-10)
            class_iou_class.append(class_iou)
            class_miou += class_iou
        class_miou = class_miou*1.0 / len(self.class_intersection_meter)


        res = {}
        res["mIoU"] = 100 * class_miou
        for i in range(self.split_gap):
            res["mIoU-{}".format(i+1)] = class_iou_class[i]

        res["FB-IoU"] = 100 * mIoU
        res["FB-mAcc"] = 100 * mAcc
        res["FB-allAcc"] = 100 * allAcc

        for i in range(2):
            res["FB-mIoU-{}".format(i)] = iou_class[i]
            res["FB-accuracy-{}".format(i)] = accuracy_class[i]


        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)


        ### reset 
        self.intersection_meter.reset()
        self.union_meter.reset()
        self.target_meter.reset()
        self.class_intersection_meter = [0]*self.split_gap
        self.class_union_meter = [0]*self.split_gap  
        ###
        
        return results
