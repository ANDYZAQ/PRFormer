split=3
dataset=pascal

# step1
# CUDA_VISIBLE_DEVICES=4,9 \
# python train.py \
# --config-file configs/${dataset}/step1_101.yaml \
# --num-gpus 2 #\
# 2>&1 | tee ./out/train.log

# Pascal 多卡
# CUDA_VISIBLE_DEVICES=6,7 \
# python train.py \
# --config-file configs/${dataset}/step2_${split}.yaml \
# --num-gpus 2 \
# --dist-url "auto" #\
# 2>&1 | tee ./out/train.log

# sleep 4h
# Pascal 单卡
CUDA_VISIBLE_DEVICES=9 \
python train.py \
--config-file configs/${dataset}/step2_${split}.yaml \
--num-gpus 1 

# Pascal_swin 单卡
# CUDA_VISIBLE_DEVICES=3 \
# python train.py \
# --config-file configs/pascal/step2_swin.yaml \
# --num-gpus 1

# COCO 单卡
# CUDA_VISIBLE_DEVICES=5 \
# python train.py \
# --config-file configs/coco/step2.yaml \
# --num-gpus 1

# Pascal 单循环
# for split in 0 1 2 3
# do
#     CUDA_VISIBLE_DEVICES=8,9 \
#     python train.py \
#     --config-file configs/${dataset}/step2_${split}.yaml \
#     --num-gpus 2 \
#     --dist-url "auto"
# done

# COCO 多卡
# CUDA_VISIBLE_DEVICES=8,9 \
# python train.py \
# --config-file configs/coco/step2_101.yaml \
# --num-gpus 2 \
# --dist-url "auto" \
# 2>&1 | tee ./out/train.log
