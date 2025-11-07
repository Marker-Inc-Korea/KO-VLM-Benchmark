# Markr-AI/Gukbap-Ovis2-16B-VL

python eval_OCRAG.py \
    --base_model [...VLM_model...] \
    --thinking_mode False \
    --huggingface_token [...your_token...] \
    --dataset_path ./data/handwritten_complex_document_OCR_benckmark.xlsx \
    --image_path ./images/document/ \
    --cutoff_len 4096
