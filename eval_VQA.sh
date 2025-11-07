python eval_VQA.py \
    --base_model [...VLM_model...] \
    --thinking_mode False \
    --huggingface_token [...huggingface_token...] \
    --dataset_path ./data/Sampled_시각화_자료_질의응답_데이터_benchmark.csv \
    --image_path ./images/VQA_images/ \
    --cutoff_len 1024
