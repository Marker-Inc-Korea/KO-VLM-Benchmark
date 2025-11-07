import torch
import pandas as pd
import fire

from huggingface_hub import login
from transformers import AutoModelForCausalLM, Gemma3ForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration

# 4.48.0
from transformers import MllamaForConditionalGeneration,MllamaProcessor
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

# 4.49.0 > // ovis2 possible

from dataset.vqa_func import VQA_19_Eval


'''
if v not in ALL_PARALLEL_STYLES:
TypeError: argument of type 'NoneType' is not iterable
->
from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

-> or transformers 4.51.3
'''

# 19_VQA
def main(
        dataset = '19_VQA',
        base_model = 'Qwen/Qwen2.5-VL-32B-Instruct',
        thinking_mode = False,
        huggingface_token = '[...your_token...]',
        dataset_path = './data/Sampled_시각화_자료_질의응답_데이터_benchmark.csv',
        image_path = './data/images',
        cutoff_len = 2048,
    ):
    
    login(token=huggingface_token)
    
    ## Model loading
    device_map = 'auto'
    
    if ('gemma-3' in base_model) or ('Gemma3' in base_model):
        print(base_model)
        model = Gemma3ForConditionalGeneration.from_pretrained( # Gemma3ForConditionalGeneration
            base_model,
            torch_dtype=torch.bfloat16,
            #cache_dir='/home/kyujin/share',
            device_map=device_map,
            attn_implementation='flash_attention_2'
        )
        
        processor = AutoProcessor.from_pretrained(base_model)

    elif 'Ovis2.5' in base_model:
        print(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device_map,
        )

    elif 'Ovis' in base_model:
        print(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device_map,
            #cache_dir='/home/kyujin/share',
            torch_dtype=torch.float16,
            trust_remote_code=True,
            multimodal_max_length=cutoff_len # 2048
        )

        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()
    
    elif 'Bllossom' in base_model:
        print(base_model)
        model = MllamaForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            #cache_dir='/home/kyujin/share',
            device_map=device_map
            )
        
        processor = MllamaProcessor.from_pretrained(base_model)
        
    elif 'VARCO' in base_model:
        print(base_model)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype="float16",
            device_map=device_map,
            #cache_dir='/home/kyujin/share',
            attn_implementation="flash_attention_2"
        )

        processor = AutoProcessor.from_pretrained(base_model, device_map=device_map)

    elif 'Qwen2.5' in base_model:
        print(base_model)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype="float16",
            device_map="auto",
            #cache_dir='/home/kyujin/share',
            attn_implementation="flash_attention_2"
        )

        processor = AutoProcessor.from_pretrained(base_model)

    elif 'Qwen3' in base_model:
        if 'B-A' in base_model: # Moe
            print(base_model)
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype="float16",
                device_map="auto",
                attn_implementation="flash_attention_2"
            )

            processor = AutoProcessor.from_pretrained(base_model)
        else:
            print(base_model)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype="float16",
                device_map="auto",
                attn_implementation="flash_attention_2"
            )

            processor = AutoProcessor.from_pretrained(base_model)
        
    else:
        raise Exception("Not implementation!!")
    
    ### Evaluation
    # 1500개
    if '19_VQA' in dataset:
        eval_dataset = pd.read_csv(dataset_path)
        print("Dataset length:", len(eval_dataset))
        
        # TODO: gemma3 not yet
        if ('Gukbap-Gemma3' in base_model) or ('gemma-3' in base_model):
            average = VQA_19_Eval(eval_dataset, model, image_path, processor, None, 'gemma3')

        elif 'Ovis2.5' in base_model:
            average = VQA_19_Eval(eval_dataset, model, image_path, None, None, 'ovis2.5', thinking_mode)
        
        elif 'Ovis' in base_model:
            average = VQA_19_Eval(eval_dataset, model, image_path, text_tokenizer, visual_tokenizer, 'ovis')
        
        elif 'Bllossom' in base_model:
            average = VQA_19_Eval(eval_dataset, model, image_path, processor, None, 'bllossom')
        
        elif 'VARCO' in base_model:
            if '2.0' in base_model:
                average = VQA_19_Eval(eval_dataset, model, image_path, processor, None, 'VARCO-2.0')
            else:
                average = VQA_19_Eval(eval_dataset, model, image_path, processor, None, 'VARCO')

        elif 'Qwen2.5' in base_model:
            average = VQA_19_Eval(eval_dataset, model, image_path, processor, None, 'qwen2.5')

        elif 'Qwen3' in base_model:
            average = VQA_19_Eval(eval_dataset, model, image_path, processor, None, 'qwen3')

        print("### 19_VQA(시각화QA질의응답데이터셋) score:", average*100)
    
    else:
        raise Exception("### Not implementation!!")


if __name__ == '__main__':
    torch.cuda.empty_cache() 
    fire.Fire(main)
