dataset=coco

# step1
CUDA_VISIBLE_DEVICES=9 \
python train.py \
--config-file configs/${dataset}/step1_101.yaml \
--num-gpus 1 

# Pascal 多卡
# CUDA_VISIBLE_DEVICES=8,9 \
# python train.py \
# --config-file configs/${dataset}/step1.yaml \
# --num-gpus 2 \
# --dist-url "auto" #\
# 2>&1 | tee ./out/train.log
