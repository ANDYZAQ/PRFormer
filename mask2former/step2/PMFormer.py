# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch, os, cv2, math
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
from ..modeling.feature_alignment.self_align import MySelfAlignLayer
from ..modeling.feature_alignment.cross_align import CrossAT
from ..modeling.transformer_decoder.position_encoding import PositionEmbeddingSine

from ..modeling.backbone import resnet as models
from ..modeling.backbone import svf
from ..modeling.backbone.PSPNet import OneModel as PSPNet
from ..modeling.backbone.ASPP import ASPP
from ..modeling.backbone.FEM import FEM


@META_ARCH_REGISTRY.register()
class PMFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

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
        # few shot
        shot: int,
        fewshot_weight: float,
        pre_norm: bool,
        conv_dim:int,
        dataset: str,
        split: int,
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
        self.info_nce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
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
        self.fewshot_weight = fewshot_weight

        pro_dim4, pro_dim3, pro_dim2 = 2048, 1024, 512
        down_dim4, down_dim3, down_dim2 = 2048, 1024, 512
        pro_dim4, down_dim4 = conv_dim, conv_dim

        self.adapters = nn.ModuleList()
        for i in range(3):
            self.adapters.append(nn.Sequential(
                nn.Linear(pro_dim4, pro_dim4 * 2), 
                nn.ReLU(), 
                nn.Dropout(0.5), 
                nn.Linear(pro_dim4*2, down_dim4)
            ))
        self.adapter3 = nn.Sequential(
            nn.Linear(pro_dim4 * 3, pro_dim4 * 6),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(pro_dim4 * 6, down_dim4 * 3)
        )
        self.union_adapter = nn.Linear(down_dim4 * 6, down_dim4 * 3)
        
        self.fc1 = nn.Linear(num_queries*2, num_queries*5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_queries*5, num_queries)

        # create the queue
        self.register_buffer("queue", torch.randn(down_dim4*3, 15 if dataset=="pascal" else 60)) # dim, num_class
        self.queue = F.normalize(self.queue, dim=0)

        # Level 2-3 line
        # pro_dim2, pro_dim3 = conv_dim, conv_dim
        fea_dim = 256
        self.down_query = nn.Sequential(
            nn.Conv2d(pro_dim2+pro_dim3, fea_dim, kernel_size=1, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Dropout2d(0.5)
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(pro_dim2+pro_dim3, fea_dim, kernel_size=1, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Dropout2d(0.5)
        )

        self.top = 10
        self.init_merge = nn.Sequential(
            nn.Conv2d(fea_dim*2 + self.top*2 + 2, fea_dim, kernel_size=1, bias=False), 
            nn.ReLU(inplace=True),
        )
        self.init_merge_3 = nn.Sequential(
            nn.Conv2d(fea_dim + self.top*2 + 2, fea_dim, kernel_size=3, padding=1, bias=False), 
            nn.ReLU(inplace=True),
        )
        self.init_merge_6 = nn.Sequential(
            nn.Conv2d(fea_dim + self.top*2 + 2, fea_dim, kernel_size=3, stride=1, padding=6,dilation=6, bias=True),
            nn.ReLU(),
        )
        
        self.res2 = nn.Sequential(
            nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=False), 
            nn.ReLU(inplace=True), 
        )
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Dropout2d(0.1), 
            nn.Conv2d(fea_dim, 2, kernel_size=1)
        )

    @classmethod
    def from_config(cls, cfg):
        if cfg.MODEL.BACKBONE.NAME == "D2SwinTransformer":
            backbone = build_backbone(cfg)
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
            # few shot
            "shot": cfg.DATASETS.SHOT,
            'fewshot_weight': cfg.MODEL.MASK_FORMER.FEWSHOT_WEIGHT,
            "pre_norm": cfg.MODEL.MASK_FORMER.PRE_NORM,
            "conv_dim": cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM, 
            "dataset": cfg.DATASETS.dataname,
            "split": cfg.DATASETS.SPLIT,
            "depth": cfg.MODEL.RESNETS.DEPTH,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        bs = len(batched_inputs)
        imgsz = batched_inputs[0]["image"].size()[-2:]
        sub_cls = [x["subcls_list"][0] for x in batched_inputs]

        query_feats = self.extract_features(batched_inputs)
        outputs, mask_features = self.gen_proposal(query_feats)
        multi_level_features = outputs['multi_scale_features'] # 4-3-2 ch:256
        query_mid = self.mid_process(query_feats, self.down_query)

        supp_label = self.trans_supp_labels(batched_inputs) # trans supp_label shape

        supp_all_feats = []
        supp_mids = []
        supp_multi_level_features = []
        for i in range(self.shot):
            supp_feats = self.extract_features(batched_inputs, idx=i)
            supp_outputs = self.gen_proposal(supp_feats, only_features=True)   # features from pixel decoder
            supp_multi_level_features.append(supp_outputs['multi_scale_features'])
            supp_mid = self.mid_process(supp_feats, self.down_supp)
            supp_mids.append(supp_mid)
            supp_all_feats.append(supp_feats)

        # Adapt query prototype to query cls
        query_cls, query_fu_cls, query_sep_cls, _ = self.adapt_multi_to_proto(multi_level_features, mask_features) # bs, 100, 256*3

        # Adapt supp prototypes to supp cls
        sup_clss = []
        sup_pros = []
        sup_pros_4 = []
        for i in range(self.shot):
            # s_label = F.interpolate(supp_label[i].unsqueeze(1).float(), size=supp_all_feats[i]['res5'].shape[-2:], mode='bilinear', align_corners=True)
            # sup_feat_4 = supp_all_feats[i]['res5'] * s_label
            sup_pro_4 = self.get_propotype(supp_all_feats[i]['res5'], supp_label[i]) # bs, 256
            sup_pros_4.append(sup_pro_4)

            sup_cls, sup_fu_cls, sup_sep_cls, supp_pro = self.adapt_multi_to_proto(supp_multi_level_features[i], supp_label[i]) # bs, 256*3
            sup_clss.append(sup_cls)
            sup_pros.append(supp_pro)

            s_label = F.interpolate(supp_label[i].unsqueeze(1).float(), size=query_mid.shape[-2:], mode='bilinear', align_corners=True)
            supp_mids[i] = supp_mids[i] * s_label
            supp_mids[i] = self.get_propotype(supp_mids[i], supp_label[i]) # bs, 256

        sup_pro = sum(sup_pros) / self.shot
        sup_cls = sum(sup_clss) / self.shot
        supp_mid = sum(supp_mids) / self.shot
        sup_pros_4 = sum(sup_pros_4) / self.shot

        prior_mask = self.generate_prior(query_feats['res5'], supp_all_feats, supp_label) # bs, 1, h, w
        prior_mask = F.interpolate(prior_mask, size=mask_features.shape[-2:], mode="bilinear", align_corners=False)
        mask_sim = mask_features * prior_mask # bs, 100, h, w
        mask_features_avg = F.avg_pool2d(mask_features, kernel_size=mask_features.shape[-2:]) # bs, 100, 1, 1
        mask_sim = F.avg_pool2d(mask_sim, kernel_size=mask_sim.shape[-2:]) / mask_features_avg # bs, 100, 1, 1

        similarities = torch.cosine_similarity(query_cls, sup_cls.unsqueeze(1), dim = 2) # bs, 100

        similarities = self.fc2(self.relu(self.fc1(torch.cat([similarities, mask_sim.squeeze(-1).squeeze(-1)], dim=1))))
        # similarities = self.fc2(self.relu(self.fc1(similarities)))

        prop_out_fg = torch.einsum("bq,bqhw->bhw", similarities, mask_features).unsqueeze(1)
        dout_bg = 1 - prop_out_fg
        prop_out = torch.cat([dout_bg, prop_out_fg],1)

        # sorting masks according to similarity
        sorted_similarities, sim_indices = torch.sort(similarities, dim=1, descending=True)
        # select top-5 and last-5 masks
        sorted_similarities = torch.cat([sorted_similarities[:, :self.top], sorted_similarities[:, -self.top:]], dim=1)
        sim_indices = torch.cat([sim_indices[:, :self.top], sim_indices[:, -self.top:]], dim=1)
        sorted_proposals = torch.gather(mask_features, dim=1, index=sim_indices.unsqueeze(-1).unsqueeze(-1).expand(bs, self.top*2, mask_features.shape[-2], mask_features.shape[-1]))



        query_mid = F.interpolate(query_mid, mask_features.shape[-2:], mode="bilinear", align_corners=True)
        supp_mid = supp_mid.unsqueeze(-1).unsqueeze(-1).expand_as(query_mid)
        weighted_proposals = sorted_similarities.unsqueeze(-1).unsqueeze(-1) * sorted_proposals
        
        merged_feat = torch.cat([query_mid, supp_mid, weighted_proposals, prop_out_fg, prior_mask], dim=1)
        mixed_feat = self.init_merge(merged_feat)
        mixed_feat = self.init_merge_3(torch.cat([mixed_feat, weighted_proposals, prop_out_fg, prior_mask], dim=1))

        mixed_feat = self.init_merge_6(torch.cat([mixed_feat, weighted_proposals, prop_out_fg, prior_mask], dim=1))

        mixed_feat = self.res2(mixed_feat) + mixed_feat
        final_out = self.cls(mixed_feat)

        prop_out =F.interpolate(prop_out, size=imgsz, mode="bilinear", align_corners=True)
        final_out =F.interpolate(final_out, size=imgsz, mode="bilinear", align_corners=True)

        if self.training:
            # label
            labels = self.get_label(batched_inputs)

            loss_nce = self.cal_info_nce(query_cls, sup_pro, similarities, sub_cls)
            loss_nce_fu = self.cal_info_nce(query_fu_cls, sup_pro, similarities, sub_cls)
            loss_nce_sep = self.cal_info_nce(query_sep_cls, sup_pro, similarities, sub_cls)
            loss_kl = self.cal_kl(similarities, mask_features, labels)

            losses = {}
            losses_for_fewshot = self.criterion_for_fewshot(final_out, labels) 
            losses_for_prop = self.criterion_for_fewshot(prop_out, labels) 
            losses["finalFSS_loss"] = losses_for_fewshot * self.fewshot_weight
            losses["propFSS_loss"] = losses_for_prop * self.fewshot_weight
            losses["info_nce_loss"] = loss_nce * self.fewshot_weight * 0.1
            losses["info_nce_mid_loss"] = (loss_nce_fu + loss_nce_sep) * self.fewshot_weight * 0.05
            losses["kl_loss"] = loss_kl * self.fewshot_weight * 2

            return losses
        else:
            processed_results = {"few_shot_result": final_out}
            processed_results["proposals"] = mask_features
            processed_results["sup_pro"] = sup_pro
            processed_results["sup_cls"] = sup_cls
            processed_results["cls"] = torch.tensor(sub_cls).to(self.device)
            processed_results["sup_pro_4"] = sup_pros_4

            return processed_results
        
    def extract_features(self, batched_inputs, idx=None):
        "Feature extraction from backbone"
        if idx is None:
            images = [x["image"].to(self.device) for x in batched_inputs]
        else:
            images = [x["support_img"][idx].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        with torch.no_grad():
            if self.backbone is None:
                feat_0 = self.layer0(images.tensor)
                feat_1 = self.layer1(feat_0)
                feat_2 = self.layer2(feat_1)
                feat_3 = self.layer3(feat_2)
                feat_4 = self.layer4(feat_3)
                features = {'res5':feat_4, 'res4':feat_3, 'res3':feat_2, 'res2':feat_1}
            else:
                features = self.backbone(images.tensor)
        return features
    
    def gen_proposal(self, features, only_features=False):
        """
            Generate proposals with features from backbone
            :param features: features from backbone
            :param only_features: if only return features
        """
        h, w = features['res3'].shape[-2:]
        feat_3_ = F.interpolate(features['res4'], size=(h//2, w//2), mode="bilinear", align_corners=True)
        feat_4_ = F.interpolate(features['res5'], size=(h//4, w//4), mode="bilinear", align_corners=True)
        prop_feats = {'res5':feat_4_, 'res4':feat_3_, 'res3':features['res3'], 'res2':features['res2']}
        with torch.no_grad():
            outputs = self.sem_seg_head(prop_feats, only_features)   # , labels
            if only_features:
                return outputs
            mask_features = outputs["pred_masks"].sigmoid()
            return outputs, mask_features
        
    def mid_process(self, features, down:nn.Sequential):
        """
            Fusion middle level features
            :param features: features from backbone
            :param down: Convolutional process for fusion
        """
        tmp_feat3 = F.interpolate(features["res4"], features["res3"].shape[-2:], mode="bilinear", align_corners=True)
        mid_feat = torch.cat([tmp_feat3, features["res3"]], dim=1)
        mid_feat = down(mid_feat)
        return mid_feat
    
    def trans_supp_labels(self, batched_inputs):
        """
        trans sup_label from bs*shot*h*w to shot*bs*h*w
        :param batched_inputs: input information
        """
        sup_label = self.get_label(batched_inputs, "support_label")
        sup_label = sup_label.permute(1, 0, 2, 3)
        return sup_label
    
    def adapt_multi_to_proto(self, multi_feats, mask_features):
        """
        Adapt multi-level features to prototype
        :param multi_feats: multi-level features
        :param mask_features: mask features
        """
        protos = []
        multi_protos = []
        for idx, feat in enumerate(multi_feats):
            proto = self.get_propotype(feat, mask_features)
            protos.append(proto)
            multi_protos.append(self.adapters[idx](proto))
        multi_proto = torch.cat(multi_protos, dim=-1)
        ori_proto = torch.cat(protos, dim=-1)
        proto = self.adapter3(ori_proto)
        union_proto = self.union_adapter(torch.cat([multi_proto, proto], dim=-1))
        return union_proto, multi_proto, proto, ori_proto # q:bs, 100, 256, s:bs, 256
    
    def weighted_prototypes(self, prototypes, weighted_soft):
        """
        Weighted prototypes
        :param prototypes: prototypes
        :param weighted_soft: weighted soft
        """
        prototypes = torch.stack(prototypes, dim=2).unsqueeze(-1) # bs, c, shot, 1
        prototypes = (weighted_soft.permute(0,2,1,3) * prototypes).sum(2)
        return prototypes.squeeze(-1) # bs, c
    
    def unify_multi_feats_size(self, multi_feats):
        """
        Unify the size of multi-level features
        :param multi_feats: multi-level features
        """
        max_size = multi_feats[-1].shape[-2:]
        for i in range(len(multi_feats)-1):
            multi_feats[i] = F.interpolate(multi_feats[i], size=max_size, mode="bilinear", align_corners=True)
        return multi_feats
    
    def unify_multi_feats(self, multi_feats):
        """
        Concat multi-level features
        :param multi_feats: multi-level features
        """
        multi_feats = self.unify_multi_feats_size(multi_feats)
        multi_feats = torch.cat(multi_feats, dim=1)
        return multi_feats

    def get_iou(self, pred, target):
        b, c, h, w = pred.shape
        target = target.unsqueeze(1)
        target = F.interpolate(target, pred.shape[-2:], mode="bilinear", align_corners=True)

        pred = pred.reshape(b, c,-1)
        target = target.reshape(b, 1, -1)
        
        #compute the IoU of the foreground
        Iand1 = torch.sum(target*pred, dim = -1)
        Ior1 = torch.sum(target, dim = -1) + torch.sum(pred, dim = -1)-Iand1 + 0.0000001
        IoU1 = Iand1/Ior1

        return IoU1
    
    def get_propotype(self, features_for_propotype, label):
        if len(label.shape) == 3:
            label = label.unsqueeze(1)
        label = F.interpolate(label, size=features_for_propotype.shape[-2:], mode='bilinear', align_corners=True)
        weight = torch.sum(label, dim=(2, 3))  # s:bs * 1 q:bs*100
        if label.size(1) != 1:
            # query for mask features
            features_for_propotype = features_for_propotype.unsqueeze(1)  # bs * 1 * 256 * h * w
            label = label.unsqueeze(2)  # bs * 100 * 1 * h * w
            weight = weight.unsqueeze(-1)  # bs * 100 * 1
        propotype = features_for_propotype * label
        propotype = torch.sum(propotype, dim=(-1, -2)) / (weight + 0.0000001)  # q:bs, 100, 256, s:bs, 256

        return propotype

    def generate_prior(self, query_feat_high, supp_feat_high, s_y):
        bsize, _, sp_sz, _ = query_feat_high.size()[:]
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for st in range(self.shot):
            # tmp_mask = (s_y[:,st,:,:] == 1).float().unsqueeze(1)
            tmp_mask = (s_y[st] == 1).float().unsqueeze(1)
            tmp_mask = F.interpolate(tmp_mask, size=(sp_sz, sp_sz), mode='bilinear', align_corners=True)

            tmp_supp_feat = supp_feat_high[st]['res5'] * tmp_mask               
            q = query_feat_high.flatten(2).transpose(-2, -1)  # [bs, h*w, c]
            s = tmp_supp_feat.flatten(2).transpose(-2, -1)  # [bs, h*w, c]

            tmp_query = q
            tmp_query = tmp_query.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s               
            tmp_supp = tmp_supp.contiguous() 
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            # corr_query = F.interpolate(corr_query, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)  
        corr_query_mask = torch.cat(corr_query_mask_list, 1)
        # corr_query_mask = (weighted_soft * corr_query_mask).sum(1,True) 
        corr_query_mask = corr_query_mask.mean(1,True)
        return corr_query_mask
    
    def get_label(self, batched_inputs, type="label"):
        label = [x[type].to(self.device) for x in batched_inputs]   # bs*h*w
        bs = len(label)
        for i in range(bs):
            label[i] = label[i].unsqueeze(0)
        label = torch.cat(label, dim = 0)
        labels = label.clone()
        labels[label == 255] = 0
        return labels
    
    def cal_info_nce(self, query_cls, sup_pro, similarities, sub_cls):
        query_cls = query_cls * similarities.unsqueeze(-1) # bs, 100, 1000
        query_cls = query_cls.transpose(1, 2).mean(dim = 2) # bs, 1000
        info_pos = torch.einsum("bc, bc->b", query_cls, sup_pro)
        info_neg = torch.einsum("bc, ck->bk", query_cls, self.queue.clone().detach())
        logits = torch.cat([info_pos.unsqueeze(1), info_neg], dim=1)
        logits = logits / 0.07
        compare_labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        loss_nce = self.info_nce(logits, compare_labels)
        # refresh queue
        self.queue[:, sub_cls] = (self.queue[:, sub_cls] + sup_pro.transpose(-1, -2))/2

        return loss_nce
    
    def cal_kl(self, similarities, mask_features, labels):
        ious = self.get_iou(mask_features, labels)
        similarities = F.log_softmax(similarities, dim = 1)
        ious = F.softmax(ious, dim = 1)
        loss_kl = self.kl(similarities, ious)
        return loss_kl