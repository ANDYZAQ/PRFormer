
import os

def add_seed(cfg):
    if cfg.DATASETS.dataname == 'pascal':
        if cfg.DATASETS.SPLIT == 0:
            seed = 4604572
        elif  cfg.DATASETS.SPLIT == 1:
            seed = 7743485
        elif  cfg.DATASETS.SPLIT == 2:
            seed = 5448843
        elif  cfg.DATASETS.SPLIT == 3:
            seed = 2534673
    if cfg.DATASETS.dataname == 'coco':
        if cfg.DATASETS.SPLIT == 0:
            seed = 8420323
        elif  cfg.DATASETS.SPLIT == 1:
            seed = 27163933
        elif  cfg.DATASETS.SPLIT == 2:
            seed = 8162312
        elif  cfg.DATASETS.SPLIT == 3:
            seed = 3391510
    if cfg.DATASETS.dataname == 'c2pv':
        seed = 321
    return ['SEED', seed]

def add_step1dir(cfg):
    root = cfg.OUTPUT_DIR if cfg.OUTPUT_DIR else os.path.join('out', cfg.DATASETS.dataname)
    OUTPUT_DIR = os.path.join(root,'step1', cfg.MODEL.META_ARCHITECTURE, str(cfg.DATASETS.SPLIT))
    return ['OUTPUT_DIR', OUTPUT_DIR]

def add_step2dir(cfg):
    root = cfg.OUTPUT_DIR if cfg.OUTPUT_DIR else os.path.join('out', cfg.DATASETS.dataname)
    OUTPUT_DIR = os.path.join(root,'step2', cfg.MODEL.META_ARCHITECTURE, f'_{cfg.DATASETS.SHOT}shots' if cfg.DATASETS.SHOT>1 else '',  str(cfg.DATASETS.SPLIT))
    return ['OUTPUT_DIR', OUTPUT_DIR]

def add_dir(cfg):
    if 'ori' in cfg.INPUT.DATASET_MAPPER_NAME:
        step = 'step1'
    elif cfg.MODEL.MASK_FORMER.FREEZE_BODY == False:
        step = 'step2'
    else:
        step = 'step3'

    root = cfg.OUTPUT_DIR if cfg.OUTPUT_DIR else os.path.join('out', cfg.DATASETS.dataname)
    OUTPUT_DIR = os.path.join(root, step, cfg.MODEL.META_ARCHITECTURE, f'_{cfg.DATASETS.SHOT}shots' if cfg.DATASETS.SHOT>1 else '', str(cfg.DATASETS.SPLIT))
    return ['OUTPUT_DIR', OUTPUT_DIR]

def add_dataset(cfg):
    DATASETS_TRAIN = (cfg.DATASETS.TRAIN[0] + str(cfg.DATASETS.SPLIT), )
    DATASETS_TEST = (cfg.DATASETS.TEST[0] + str(cfg.DATASETS.SPLIT) +'_'+ str(cfg.DATASETS.SHOT) + 'shot',)
    return ['DATASETS.TRAIN', DATASETS_TRAIN, 'DATASETS.TEST', DATASETS_TEST]