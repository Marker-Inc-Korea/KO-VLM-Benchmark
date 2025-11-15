import fire
import pandas as pd
import torch
from huggingface_hub import login

# 4.48.0
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
    MllamaForConditionalGeneration,
    MllamaProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

# 4.49.0 > // ovis2 possible
from ko_vlm_benchmark.dataset.vdc_func import VDC_Eval

"""
if v not in ALL_PARALLEL_STYLES:
TypeError: argument of type 'NoneType' is not iterable
->
from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

-> or transformers 4.51.3
"""


# 19_VQA
def main(
    dataset="VDC",
    base_model="Markr-AI/Gukbap-Gemma3-12B-VL",
    thinking_mode=False,
    huggingface_token: str | None = None,
    dataset_path="./data/Gemini_sampled_멀티모달_정보검색_데이터_benchmark_final.csv",
    image_path="./data/images",
    cutoff_len=2048,
):

    if huggingface_token is not None:
        login(token=huggingface_token)

    ## Model loading
    device_map = "auto"

    if ("gemma-3" in base_model) or ("Gemma3" in base_model):
        print(base_model)
        model = Gemma3ForConditionalGeneration.from_pretrained(  # Gemma3ForConditionalGeneration
            base_model, torch_dtype=torch.bfloat16, device_map=device_map, attn_implementation="flash_attention_2"
        )

        processor = AutoProcessor.from_pretrained(base_model)

    elif "Ovis2.5" in base_model:
        print(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device_map,
        )

    elif ("Ovis" in base_model) or ("Gukbap" in base_model):
        print(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            multimodal_max_length=cutoff_len,
        )

        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()

    elif "Bllossom" in base_model:
        print(base_model)
        model = MllamaForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map=device_map
        )

        processor = MllamaProcessor.from_pretrained(base_model)

    elif "VARCO" in base_model:
        print(base_model)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            base_model, torch_dtype="float16", device_map=device_map, attn_implementation="flash_attention_2"
        )

        processor = AutoProcessor.from_pretrained(base_model, device_map=device_map)

    elif "Qwen2.5" in base_model:
        print(base_model)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model, torch_dtype="float16", device_map="auto", attn_implementation="flash_attention_2"
        )

        processor = AutoProcessor.from_pretrained(base_model)

    elif "Qwen3" in base_model:
        if "B-A" in base_model:  # Moe
            print(base_model)
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                base_model, torch_dtype="float16", device_map="auto", attn_implementation="flash_attention_2"
            )

            processor = AutoProcessor.from_pretrained(base_model)
        else:
            print(base_model)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model, torch_dtype="float16", device_map="auto", attn_implementation="flash_attention_2"
            )

            processor = AutoProcessor.from_pretrained(base_model)

    else:
        raise NotImplementedError("Not implementation!!")

    ### Evaluation
    if "VDC" in dataset:
        eval_dataset = pd.read_csv(dataset_path, encoding="cp949")
        # eval_dataset = eval_dataset.iloc[:100]
        print("Dataset length:", len(eval_dataset))

        # TODO: gemma3 not yet
        if ("Gukbap-Gemma3" in base_model) or ("gemma-3" in base_model):
            average, type1_avg, type2_avg = VDC_Eval(eval_dataset, model, image_path, processor, None, "gemma3")

        elif "Ovis2.5" in base_model:
            average, type1_avg, type2_avg = VDC_Eval(
                eval_dataset, model, image_path, None, None, "ovis2.5", thinking_mode
            )

        elif ("Ovis" in base_model) or ("Gukbap" in base_model):
            average, type1_avg, type2_avg = VDC_Eval(
                eval_dataset, model, image_path, text_tokenizer, visual_tokenizer, "ovis"
            )

        elif "Bllossom" in base_model:
            average, type1_avg, type2_avg = VDC_Eval(eval_dataset, model, image_path, processor, None, "bllossom")

        elif "VARCO" in base_model:
            if "2.0" in base_model:
                average, type1_avg, type2_avg = VDC_Eval(eval_dataset, model, image_path, processor, None, "VARCO-2.0")
            else:
                average, type1_avg, type2_avg = VDC_Eval(eval_dataset, model, image_path, processor, None, "VARCO")

        elif "Qwen2.5" in base_model:
            average, type1_avg, type2_avg = VDC_Eval(eval_dataset, model, image_path, processor, None, "qwen2.5")

        elif "Qwen3" in base_model:
            average, type1_avg, type2_avg = VDC_Eval(eval_dataset, model, image_path, processor, None, "qwen3")

        """
        elif 'GPT' in base_model:
            pass
        """

        print("### 멀티모달정보검색 전체 score:", average * 100)
        print("### 멀티모달정보검색 보고서 score:", type1_avg * 100)
        print("### 멀티모달정보검색 보도자료 score:", type2_avg * 100)

    else:
        raise NotImplementedError("### Not implementation!!")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(main)
