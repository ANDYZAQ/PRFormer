from contextlib import ExitStack, contextmanager
import datetime
import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from collections import OrderedDict, abc
import copy
import itertools
import time
from typing import Any, Dict, List, Set
import cv2
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import numpy as np

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    # build_detection_test_loader,
    # build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
# from detectron2.solver import build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.events import EventStorage
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.utils.logger import setup_logger, log_every_n_seconds
from detectron2.utils.visualizer import Visualizer

from mask2former.utils.addcfg import *
from mask2former import (
    FewShotSemSegEvaluator,  ###
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)
from mask2former.data import (
    FewShotDatasetMapper_stage2,
    FewShotDatasetMapper_stage1,
    FewShotDatasetMapper_stage2_change, 
    # FewShotVideoDatasetMapper,
    build_detection_train_loader,
    build_detection_test_loader,
)

logger = logging.getLogger("detectron2")

def build_optimizer(cfg, model):
    weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
    weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

    defaults = {}
    defaults["lr"] = cfg.SOLVER.BASE_LR
    defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if "backbone" in module_name:
                hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
            if (
                "relative_position_bias_table" in module_param_name
                or "absolute_pos_embed" in module_param_name
            ):
                print(module_param_name)
                hyperparams["weight_decay"] = 0.0
            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = weight_decay_norm
            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = weight_decay_embed
            params.append({"params": [value], **hyperparams})

    def maybe_add_full_model_gradient_clipping(optim):
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED
            and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
            and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        )
    elif optimizer_type == "ADAMW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params, cfg.SOLVER.BASE_LR
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer

imgsv_idx = 0
def visualization(cfg, inputs, outputs, mIoUs):
    images = [x["image"] for x in inputs]
    labels = [(x["label"]==1).int() for x in inputs] 
    predictions = outputs["few_shot_result"].argmax(dim=1)
    mask_features = outputs["proposals"]
    global imgsv_idx

    for image, label, prediction, single_mask_features, mIoU in zip(images, labels, predictions, mask_features, mIoUs):
        rootpath = os.path.join(cfg.OUTPUT_DIR, "figures", f"{mIoU:.3f}_{imgsv_idx}")
        if os.path.exists(rootpath) == False:
            os.makedirs(rootpath)

        v = Visualizer(image.permute(1, 2, 0).cpu().numpy())
        vis = v.draw_binary_mask(label.cpu().numpy(), color='fuchsia')
        vis.save(os.path.join(rootpath, f'GT_{imgsv_idx}.png'))

        v = Visualizer(image.permute(1, 2, 0).cpu().numpy())
        vis = v.draw_binary_mask(prediction.cpu().numpy(), color='blue')
        vis.save(os.path.join(rootpath, f'Pred_{imgsv_idx}.png'))

        # Save image and label
        image = cv2.cvtColor(image.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(rootpath, f'Image_{imgsv_idx}.png'), image)
        cv2.imwrite(os.path.join(rootpath, f'Label_{imgsv_idx}.png'), label.cpu().numpy() * 255)

        # for idx, proposal in enumerate(single_mask_features):
        #     proposal = proposal * 255
        #     cv2.imwrite(os.path.join(rootpath, f'Prop_{imgsv_idx}_{idx}.png'), proposal.cpu().numpy())

        # prior_mask = prior_mask * 255
        # cv2.imwrite(os.path.join(rootpath, f'Prior_{imgsv_idx}.png'), prior_mask.squeeze(0).cpu().numpy())

        imgsv_idx += 1
    

def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each
    builtin dataset. For your own dataset, you can simply create an
    evaluator manually in your script and do not have to worry about the
    hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    # semantic segmentation
    if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
        evaluator_list.append(
            FewShotSemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
                post_process_func=dense_crf_post_process
                if cfg.TEST.DENSE_CRF
                else None,
                dataname = cfg.DATASETS.dataname,
                split = cfg.DATASETS.SPLIT,
            )
        )

    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model, data_loaders, evaluators):
    results = OrderedDict()
    pro_info = {
        'prototypes': [], 
        'category': [], 
    }
    cls_info = {
        'prototypes': [], 
        'category': [], 
    }
    pro_4_info = {
        'prototypes': [], 
        'category': [], 
    }
    for dataset_name, data_loader, evaluator in zip(cfg.DATASETS.TEST, data_loaders, evaluators):
        # results_i = inference_on_dataset(model, data_loader, evaluator)
        num_devices = comm.get_world_size()
        logger = logging.getLogger(__name__)
        logger.info("Start inference on {} batches".format(len(data_loader)))

        total = len(data_loader)  # inference data loader must have a fixed length
        if evaluator is None:
            # create a no-op evaluator
            evaluator = DatasetEvaluators([])
        if isinstance(evaluator, abc.MutableSequence):
            evaluator = DatasetEvaluators(evaluator)
        evaluator.reset()

        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_data_time = 0
        total_compute_time = 0
        total_eval_time = 0
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            start_data_time = time.perf_counter()
            for idx, inputs in enumerate(data_loader):
                total_data_time += time.perf_counter() - start_data_time
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()
                outputs = model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                start_eval_time = time.perf_counter()
                mIoUs = evaluator.process(inputs, outputs)
                total_eval_time += time.perf_counter() - start_eval_time

                # visualization(cfg, inputs, outputs, mIoUs)
                pro = outputs["sup_pro"]
                cls = outputs["sup_cls"]
                cls_id = outputs["cls"]
                pro_4 = outputs["sup_pro_4"]
                pro_info["prototypes"].append(pro)
                pro_info["category"].append(cls_id)
                cls_info["prototypes"].append(cls)
                cls_info["category"].append(cls_id)
                pro_4_info["prototypes"].append(pro_4)
                pro_4_info["category"].append(cls_id)

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                    eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        (
                            f"Inference done {idx + 1}/{total}. "
                            f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                            f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                            f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                            f"Total: {total_seconds_per_iter:.4f} s/iter. "
                            f"ETA={eta}"
                        ),
                        n=5,
                    )
                start_data_time = time.perf_counter()

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
            )
        )

        results_i = evaluator.evaluate()
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)

            pro_info["prototypes"] = torch.cat(pro_info["prototypes"], dim=0).cpu().numpy()
            pro_info["category"] = torch.cat(pro_info["category"], dim=0).cpu().numpy()
            cls_info["prototypes"] = torch.cat(cls_info["prototypes"], dim=0).cpu().numpy()
            cls_info["category"] = torch.cat(cls_info["category"], dim=0).cpu().numpy()
            pro_4_info["prototypes"] = torch.cat(pro_4_info["prototypes"], dim=0).cpu().numpy()
            pro_4_info["category"] = torch.cat(pro_4_info["category"], dim=0).cpu().numpy()

            np.save(os.path.join(cfg.OUTPUT_DIR, "pro_info.npy"), pro_info["prototypes"])
            np.save(os.path.join(cfg.OUTPUT_DIR, "cls_info.npy"), cls_info["prototypes"])
            np.save(os.path.join(cfg.OUTPUT_DIR, "pro_category.npy"), pro_info["category"])
            np.save(os.path.join(cfg.OUTPUT_DIR, "pro_4_info.npy"), pro_4_info["prototypes"])


    if len(results) == 1:
        results = list(results.values())[0]
    return results

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

def build_train_loader(cfg):
    if cfg.INPUT.DATASET_MAPPER_NAME == "fewshot_sem":
        mapper = FewShotDatasetMapper_stage2(cfg, True)
        # print(build_detection_train_loader(cfg, mapper=mapper)) FewShotDatasetVideoMapper
    elif cfg.INPUT.DATASET_MAPPER_NAME == "fewshot_sem_ori":
        mapper = FewShotDatasetMapper_stage1(cfg, True) 
    else:
        mapper = None
    return build_detection_train_loader(cfg, mapper=mapper)
# return build_detection_train_loader(cfg, mapper=mapper)

def do_train(cfg, model, resume=False, data_loaders = None, evaluators = None):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    # data_loader = build_detection_train_loader(cfg)
    data_loader = build_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))

    best_mIoU = 0

    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                # and iteration != max_iter - 1
            ):
                results = do_test(cfg, model, data_loaders, evaluators)

                results_dict = dict(results)['sem_seg']
                storage.put_scalars(**results_dict, smoothing_hint=False)
                comm.synchronize()
                
                if results:   
                    cur_mIoU = results['sem_seg']['mIoU'] if 'mIoU' in results['sem_seg'] else results['sem_seg']['IoU']
                    if cur_mIoU > best_mIoU:
                        best_mIoU = cur_mIoU
                        periodic_checkpointer.save('model_best')

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts) # ['SEED', k]
    # 
    cfg.merge_from_list(add_seed(cfg))

    DATASETS_TRAIN = (cfg.DATASETS.TRAIN[0] + str(cfg.DATASETS.SPLIT), )
    DATASETS_TEST = (cfg.DATASETS.TEST[0] + str(cfg.DATASETS.SPLIT) +'_'+ str(cfg.DATASETS.SHOT) + 'shot',)
    # 
    print( 'DATASETS.TRAIN', DATASETS_TRAIN, 'DATASETS.TEST', DATASETS_TEST)
    cfg.merge_from_list(['DATASETS.TRAIN', DATASETS_TRAIN, 'DATASETS.TEST', DATASETS_TEST]) # ['SEED', k]
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    new_params = model.state_dict()
    # for i in new_params:
    #     if 'backbone' in i:
    #         print(i)
    if cfg.MODEL.WEIGHTS_ is not None:
        saved_state_dict = torch.load(cfg.MODEL.WEIGHTS_)['model']
        new_params = model.state_dict()

        for i in saved_state_dict:
            if i in new_params.keys():
                print('\t' + i)
                new_params[i] = saved_state_dict[i]

        model.load_state_dict(new_params)

    # build test set first
    data_loaders, evaluators = [], []
    for dataset_name in cfg.DATASETS.TEST:
        if cfg.INPUT.DATASET_MAPPER_NAME == "fewshot_sem":
            mapper = FewShotDatasetMapper_stage2(cfg, False)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "fewshot_sem_ori":
            mapper = FewShotDatasetMapper_stage1(cfg, False)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "fewshot_sem_change":
            mapper = FewShotDatasetMapper_stage2_change(cfg, False)
        else:
            mapper = None
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper = mapper)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        data_loaders.append(data_loader)
        evaluators.append(evaluator)

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model, data_loaders, evaluators)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
        )

    do_train(cfg, model, resume=args.resume, data_loaders=data_loaders, evaluators=evaluators)
    # return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )