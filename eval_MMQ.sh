# Markr-AI/Gukbap-Ovis2-16B-VL

python eval_MMQ.py \
    --base_model AIDC-AI/Ovis2.5-2B \
    --thinking_mode \
    --huggingface_token [...your_token...] \
    --dataset_path ./data/Gemini_sampled_멀티모달_정보검색_데이터_benchmark_final.csv \
    --image_path ./data/images/ \
    --cutoff_len 2048