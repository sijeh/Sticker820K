# export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:StickerCLIP/sticker_clip

data_root=EmojiData

save_dir=StickerCLIP/work_dir/feats

img_id_url="StickerData/jsonl/img_urls.json"
batch_size=128
context_length=64
gpu=1

resume="StickerCLIP/work_dir/checkpoints/epoch_latest.pt"

vision_model=ViT-B-16
text_model=RoBERTa-wwm-ext-base-chinese

python -u StickerCLIP/sticker_clip/eval/extract_emoji_features.py \
    --data-root=${data_root} \
    --save-dir=${save_dir} \
    --img-id-url=${img_id_url} \
    --batch-size=${batch_size} \
    --context-length=${context_length} \
    --resume=${resume} \
    --vision-model=${vision_model} \
    --text-model=${text_model} \
    --gpu=${gpu}
