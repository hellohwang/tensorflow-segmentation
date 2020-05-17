# model definition
BACKBONE="mobilenet_v2"
MODEL="depv3plus"
# hyper parameter
TOTAL_EPOCHES=50
START_EPOCHES=$((${TOTAL_EPOCHES} * 7 / 10))
BATCH_SIZE=2
CROP_SIZE=384
BASE_SIZE=512
LR=0.01

export CUDA_VISIBLE_DEVICES="6,7"

python train_gpus.py \
    --dataset="clothes" \
    --lr="${LR}" \
    --batch-size="${BATCH_SIZE}" \
    --crop-size="${CROP_SIZE}" \
    --base-size="${BASE_SIZE}" \
    --epochs="${TOTAL_EPOCHES}" \
