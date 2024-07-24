GPU=3
dataset=coco

CUDA_VISIBLE_DEVICES=${GPU} python test_viz.py \
    --config-file configs/${dataset}/eval.yaml \
    --num-gpus 1 \
    --eval-only