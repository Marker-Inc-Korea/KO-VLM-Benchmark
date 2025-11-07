# Markr-AI/Gukbap-Ovis2-16B-VL

python eval_VDC.py \
    --base_model [...VLM_model...] \
    --thinking_mode False \
    --huggingface_token [...your_token...] \
    --dataset_path ./data/Gemini_sampled_멀티모달_정보검색_데이터_benchmark_final.csv \
    --image_path ./images/VDC_images/ \
    --cutoff_len 1024
