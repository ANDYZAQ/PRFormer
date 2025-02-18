# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch, os, cv2
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import ShapeSpec


from ..modeling.criterion import SetCriterion
from ..modeling.matcher import HungarianMatcher
from ..modeling.fewshot_loss import WeightedDiceLoss
from ..modeling.backbone import resnet as models
from ..modeling.backbone import svf


@META_ARCH_REGISTRY.register()
class POSTK5(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference

        shot: int,
        fewshot_weight: float,
        depth: int, 
    ):

        super().__init__()
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = None
            resnet = models.resnet50(pretrained=True) if depth == 50 else models.resnet101(pretrained=True)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.criterion_for_fewshot = WeightedDiceLoss()   #####
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.shot = shot
        # self.fewshot_weight = fewshot_weight
        self.fewshot_weight = 5

    @classmethod
    def from_config(cls, cfg):
        if cfg.MODEL.BACKBONE.NAME == "D2SwinTransformer":
            backbone = build_backbone(cfg)
            output_shape = backbone.output_shape()
        else:
            backbone = None
            output_shape = {
                'res2': ShapeSpec(channels=256, stride=4),
                'res3': ShapeSpec(channels=512, stride=8),
                'res4': ShapeSpec(channels=1024, stride=16),
                'res5': ShapeSpec(channels=2048, stride=32),
            }
        sem_seg_head = build_sem_seg_head(cfg, output_shape)

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "shot": cfg.DATASETS.SHOT,
            'fewshot_weight': cfg.MODEL.MASK_FORMER.FEWSHOT_WEIGHT,
            "depth": cfg.MODEL.RESNETS.DEPTH,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                training:
                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                testing:
                * "few_shot_result"
                     A Tensor that represents the forground score and background score.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        with torch.no_grad():   
            if self.backbone is not None:
                features = self.backbone(images.tensor)
            else:
                query_feat_0 = self.layer0(images.tensor)
                query_feat_1 = self.layer1(query_feat_0)
                query_feat_2 = self.layer2(query_feat_1)
                query_feat_3 = self.layer3(query_feat_2)
                query_feat_4 = self.layer4(query_feat_3)
                # print(query_feat_4.shape, query_feat_3.shape, query_feat_2.shape)
                h, w = query_feat_2.shape[-2:]
                query_feat_3_ = F.interpolate(query_feat_3.clone(), size=(h//2, w//2), mode="bilinear", align_corners=True)
                query_feat_4_ = F.interpolate(query_feat_4.clone(), size=(h//4, w//4), mode="bilinear", align_corners=True)
                features = {'res5':query_feat_4_, 'res4':query_feat_3_, 'res3':query_feat_2, 'res2':query_feat_1}
        outputs = self.sem_seg_head(features)
        mask_pred_results = outputs["pred_masks"].sigmoid()
        
        # label
        label = [x["label"].to(self.device) for x in batched_inputs]   # bs*h*w
        label = torch.stack(label, dim = 0)
        labels = label.clone()
        labels[label == 255] = 0

        ious = self.get_iou(mask_pred_results, labels)
        _, index = ious.topk(5, dim=1, largest=True, sorted=True)

        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),  # h * 2  !!!
            mode="bilinear",
            align_corners=False,
        )

        bs_idx = torch.arange(0, labels.shape[0]).long()

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            for idx, idx_prop in enumerate(index.transpose(0,1)):
                out = mask_pred_results[(bs_idx, idx_prop)].unsqueeze(1)
                out_all = torch.cat([1-out, out],1)
                losses[f"FSS_loss{idx}"] = self.criterion_for_fewshot(out_all, labels) * self.fewshot_weight
            
            return losses
            
        else:
            out = mask_pred_results[(bs_idx, index[:,1])].unsqueeze(1)
            # out2 = mask_pred_results[(bs_idx, index[:,2])].unsqueeze(1)
            # out = torch.cat([out1, out2], dim = 1).mean(dim = 1).unsqueeze(1)
            dout_bg = 1 - out
            out_all = torch.cat([dout_bg, out],1)

            processed_results = {"few_shot_result": out_all}
 
            return processed_results

    def prepare_targets(self, targets, images):
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            # print(targets_per_image.gt_classes, 'targets_per_image.gt_classes')
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def get_iou(self, pred, target):
        # pred = pred.sigmoid() 
        b, c, h, w = pred.shape
        target = target.unsqueeze(1)
        # print(pred.shape, target.shape)
        # assert pred.shape == target.shape
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
            pred,
            size=(target.shape[-2], target.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )


        pred = pred.reshape(b, c,-1)
        target = target.reshape(b, 1, -1)
        
        #compute the IoU of the foreground
        Iand1 = torch.sum(target*pred, dim = -1)
        Ior1 = torch.sum(target, dim = -1) + torch.sum(pred, dim = -1)-Iand1 + 0.0000001
        IoU1 = Iand1/Ior1

        return IoU1