split=2
dataset=pascal

# Pascal 多卡
# CUDA_VISIBLE_DEVICES=0,1 \
# python train.py \
# --config-file configs/${dataset}/step2_101_${split}.yaml \
# --num-gpus 2 \
# --dist-url "auto" #\
# 2>&1 | tee ./out/train.log

# Pascal 单卡
# CUDA_VISIBLE_DEVICES=6 \
# python train.py \
# --config-file configs/${dataset}/step2_${split}.yaml \
# --num-gpus 1 #\
# --resume MODEL.WEIGHTS out/pascal/step2/PMFormer_res_iou/1/model_best.pth

# COCO 单卡
# CUDA_VISIBLE_DEVICES=2 \
# python train.py \
# --config-file configs/coco/step2_101.yaml \
# --num-gpus 1 

# COCO 多卡
CUDA_VISIBLE_DEVICES=4,5 \
python train.py \
--config-file configs/coco/step2_101.yaml \
--num-gpus 2 \
--dist-url "auto"

# Pascal Swin 单卡
# CUDA_VISIBLE_DEVICES=4 \
# python train.py \
# --config-file configs/pascal/step2_swin.yaml \
# --num-gpus 1 