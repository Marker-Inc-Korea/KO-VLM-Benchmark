import torch
import os
import re
import random
import pandas as pd

from tqdm import tqdm
from PIL import Image
from io import BytesIO

from evaluate import load
from rouge_score import rouge_scorer

from qwen_vl_utils import process_vision_info

#from llava.mm_utils import tokenizer_image_token, process_images

def check_options(options):
    
    options_prompt = []
    for i in range(len(options)):
        
        idx = options.index[i]
        row = options[i]
        #print(idx, row)
        
        if row is not None:
            prompt = idx + '. ' + row.strip()
            options_prompt.append(prompt)
    
    return options_prompt

def read_image(bytes):
    try:
        image = Image.open(BytesIO(bytes))
        return image
    
    except Exception as e:
        raise Exception(e)


def OCRAG_Eval(eval_dataset, 
                model,
                dataset_path,
                text_tokenizer, 
                visual_tokenizer,
                category,
                thinking_mode=False):
    
    all_wer_score = 0
    all_cer_score = 0
    all_rouge_score = 0
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    print(eval_dataset.columns)

    text_sample = pd.DataFrame(columns=['id', 'type', 'Modified_image_path', 'text_input'])

    print("## load metric")
    # WER
    wer_metric = load("wer")
    # CER
    cer_metric = load("cer")

    # TODO: eval code
    if category == 'gemma3':
        
        domain_list = os.listdir(dataset_path)
        
        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].Dataset_type.strip()
            gt_caption = eval_dataset.loc[i].RAG_Parsing.strip()
            img_path = eval_dataset.loc[i].image.strip()

            ## image load
            img_path = dataset_path + "/" + img_path
            image = Image.open(img_path)
            
            ## prompt
            query = '''당신은 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다. 
다음 주어진 문서에 나타난 한국어 텍스트 문단을 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
이때 텍스트는 그대로 적고, 이미지/도식은 [image]~[/image]라는 구분기호와 적절한 설명으로 대체하여 적어야합니다. 
OCR 결과:'''
            print(query)
            
            # templates
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": img_path},
                        {"type": "text", "text": query}
                    ]
                }
            ]

            ## inference
            inputs = text_tokenizer.apply_chat_template(messages, 
                                                    tokenize=True, 
                                                    return_dict=True, 
                                                    add_generation_prompt=True,
                                                    return_tensors='pt').to("cuda")
            input_len = inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=4096, do_sample=False, temperature=None)
                generation = generation[0][input_len:]

            output = text_tokenizer.decode(generation, skip_special_tokens=True)
            print(f'Output: {output}')
            
            ## TODO: calculate accuracy
            pred_caption = output.strip() # alphabet
            ref_texts = [gt_caption]
            hyp_texts = [pred_caption]

            wer_score = wer_metric.compute(references=ref_texts, predictions=hyp_texts)
            cer_score = cer_metric.compute(references=ref_texts, predictions=hyp_texts)

            if wer_score > 1:
                wer_score = 1
            if cer_score > 1:
                cer_score = 1

            print("WER:", (1-wer_score)*100)
            print("CER:", (1-cer_score)*100)

            all_wer_score += (1-wer_score)*100
            all_cer_score += (1-cer_score)*100

            rouge_scores = scorer.score(ref_texts[0], hyp_texts[0])
            print("ROUGE:", rouge_scores)
            rouge_2_score = rouge_scores['rouge2'].recall

            all_rouge_score += rouge_2_score*100

    elif category == 'ovis':

        domain_list = os.listdir(dataset_path)
        
        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].Dataset_type.strip()
            gt_caption = eval_dataset.loc[i].RAG_Parsing.strip()
            img_path = eval_dataset.loc[i].image.strip()

            ## image load
            img_path = dataset_path + "/" + img_path
            image = Image.open(img_path)
            
            ## prompt
            query = '''당신은 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다. 
다음 주어진 문서에 나타난 한국어 텍스트 문단을 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
이때 텍스트는 그대로 적고, 이미지/도식은 [image]~[/image]라는 구분기호와 적절한 설명으로 대체하여 적어야합니다. 
OCR 결과:'''
            print(query)

            ## format conversation
            prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
            attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=model.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
            pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]
            
            ## generate output
            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=4096,
                    do_sample=False,
                    top_p=None,
                    top_k=None,
                    temperature=None,
                    repetition_penalty=None,
                    eos_token_id=model.generation_config.eos_token_id,
                    pad_token_id=text_tokenizer.pad_token_id,
                    use_cache=True
                )
                
                output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
                output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
                print(f'Output: {output}')

            
            ## TODO: calculate accuracy
            pred_caption = output.strip() # alphabet
            ref_texts = [gt_caption]
            hyp_texts = [pred_caption]

            wer_score = wer_metric.compute(references=ref_texts, predictions=hyp_texts)
            cer_score = cer_metric.compute(references=ref_texts, predictions=hyp_texts)

            if wer_score > 1:
                wer_score = 1
            if cer_score > 1:
                cer_score = 1

            print("WER:", (1-wer_score)*100)
            print("CER:", (1-cer_score)*100)

            all_wer_score += (1-wer_score)*100
            all_cer_score += (1-cer_score)*100

            rouge_scores = scorer.score(ref_texts[0], hyp_texts[0])
            print("ROUGE:", rouge_scores)
            rouge_2_score = rouge_scores['rouge2'].recall

            all_rouge_score += rouge_2_score*100


    elif category == 'bllossom':
        domain_list = os.listdir(dataset_path)
        
        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].Dataset_type.strip()
            gt_caption = eval_dataset.loc[i].RAG_Parsing.strip()
            img_path = eval_dataset.loc[i].image.strip()

            ## image load
            img_path = dataset_path + "/" + img_path
            image = Image.open(img_path)
            
            ## prompt
            query = '''당신은 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다. 
다음 주어진 문서에 나타난 한국어 텍스트 문단을 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
이때 텍스트는 그대로 적고, 이미지/도식은 [image]~[/image]라는 구분기호와 적절한 설명으로 대체하여 적어야합니다. 
OCR 결과:'''
            print(query)
            
            messages = [
                {'role': 'user','content': [
                    {'type':'image'},
                    {'type': 'text','text': query}
                    ]},
                ]

            ## inference
            input_text = text_tokenizer.apply_chat_template(messages,
                                                            tokenize=False,
                                                            add_generation_prompt=True)
            
            inputs = text_tokenizer(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(model.device)
            
            ## generate output
            output = model.generate(**inputs, 
                                    max_new_tokens=4096,
                                    temperature=None,
                                    eos_token_id=text_tokenizer.tokenizer.convert_tokens_to_ids('<|eot_id|>'),
                                    use_cache=True) # If False, 60 hours
            #print(text_tokenizer.decode(output[0]))
            
            output = text_tokenizer.decode(output[0])[len(input_text):].strip()

            print(f'Output: {output}')
            
            ## TODO: calculate accuracy
            pred_caption = output.strip() # alphabet
            ref_texts = [gt_caption]
            hyp_texts = [pred_caption]

            wer_score = wer_metric.compute(references=ref_texts, predictions=hyp_texts)
            cer_score = cer_metric.compute(references=ref_texts, predictions=hyp_texts)

            if wer_score > 1:
                wer_score = 1
            if cer_score > 1:
                cer_score = 1

            print("WER:", (1-wer_score)*100)
            print("CER:", (1-cer_score)*100)

            all_wer_score += (1-wer_score)*100
            all_cer_score += (1-cer_score)*100
            
            rouge_scores = scorer.score(ref_texts[0], hyp_texts[0])
            print("ROUGE:", rouge_scores)
            rouge_2_score = rouge_scores['rouge2'].recall

            all_rouge_score += rouge_2_score*100

    elif category == 'VARCO-2.0':
        domain_list = os.listdir(dataset_path)

        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].Dataset_type.strip()
            gt_caption = eval_dataset.loc[i].RAG_Parsing.strip()
            img_path = eval_dataset.loc[i].image.strip()

            ## image load
            img_path = dataset_path + "/" + img_path
            image = Image.open(img_path)
            
            ## prompt
            text = '''당신은 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다. 
다음 주어진 문서에 나타난 한국어 텍스트 문단을 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
이때 텍스트는 그대로 적고, 이미지/도식은 [image]~[/image]라는 구분기호와 적절한 설명으로 대체하여 적어야합니다. 
OCR 결과:'''
            print(text)

            query = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": text},
                    ],
                },
            ]

            inputs = text_tokenizer.apply_chat_template(
                query,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, torch.float16)

            ## generation
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=None,
                    max_new_tokens=4096,
                    use_cache=True,
                )
            generate_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            output = text_tokenizer.decode(generate_ids_trimmed[0], skip_special_tokens=True)

            print(f'Output:\n{output}')

            ## TODO: calculate accuracy
            pred_caption = output.strip() # alphabet
            ref_texts = [gt_caption]
            hyp_texts = [pred_caption]

            wer_score = wer_metric.compute(references=ref_texts, predictions=hyp_texts)
            cer_score = cer_metric.compute(references=ref_texts, predictions=hyp_texts)

            if wer_score > 1:
                wer_score = 1
            if cer_score > 1:
                cer_score = 1

            print("WER:", (1-wer_score)*100)
            print("CER:", (1-cer_score)*100)

            all_wer_score += (1-wer_score)*100
            all_cer_score += (1-cer_score)*100

            rouge_scores = scorer.score(ref_texts[0], hyp_texts[0])
            print("ROUGE:", rouge_scores)
            rouge_2_score = rouge_scores['rouge2'].recall

            all_rouge_score += rouge_2_score*100
    
    elif category == 'VARCO':
        domain_list = os.listdir(dataset_path)

        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].Dataset_type.strip()
            gt_caption = eval_dataset.loc[i].RAG_Parsing.strip()
            img_path = eval_dataset.loc[i].image.strip()

            ## image load
            img_path = dataset_path + "/" + img_path
            image = Image.open(img_path)
            
            ## prompt
            text = '''당신은 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다. 
다음 주어진 문서에 나타난 한국어 텍스트 문단을 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
이때 텍스트는 그대로 적고, 이미지/도식은 [image]~[/image]라는 구분기호와 적절한 설명으로 대체하여 적어야합니다. 
OCR 결과:'''
            print(text)

            query = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image"},
                    ],
                },
            ]

            ## preprocessing
            prompt = text_tokenizer.apply_chat_template(query, add_generation_prompt=True)
            
            EOS_TOKEN = "<|im_end|>"
            inputs = text_tokenizer(images=image, text=prompt, return_tensors='pt').to(torch.float16).to(model.device)
            
            ## generation
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=None,
                    max_new_tokens=4096,
                    use_cache=True,
                )

            output = text_tokenizer.batch_decode(output_ids[0][inputs.input_ids.shape[1]:])
            output = ''.join(output).strip()
            if output.endswith(EOS_TOKEN):
                output = output[: -len(EOS_TOKEN)]

            print(f'Output:\n{output}')

            ## TODO: calculate accuracy
            pred_caption = output.strip() # alphabet
            ref_texts = [gt_caption]
            hyp_texts = [pred_caption]

            wer_score = wer_metric.compute(references=ref_texts, predictions=hyp_texts)
            cer_score = cer_metric.compute(references=ref_texts, predictions=hyp_texts)

            if wer_score > 1:
                wer_score = 1
            if cer_score > 1:
                cer_score = 1

            print("WER:", (1-wer_score)*100)
            print("CER:", (1-cer_score)*100)

            all_wer_score += (1-wer_score)*100
            all_cer_score += (1-cer_score)*100
            
            rouge_scores = scorer.score(ref_texts[0], hyp_texts[0])
            print("ROUGE:", rouge_scores)
            rouge_2_score = rouge_scores['rouge2'].recall

            all_rouge_score += rouge_2_score*100
            
    
    # TODO: make qwen2.5 eval pipeline
    elif category == 'qwen2.5':
        domain_list = os.listdir(dataset_path)
        #print(eval_dataset.columns)

        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].Dataset_type.strip()
            gt_caption = eval_dataset.loc[i].RAG_Parsing.strip()
            img_path = eval_dataset.loc[i].image.strip()

            ## image load
            img_path = dataset_path + "/" + img_path
            
            ## prompt
            query = '''당신은 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다. 
다음 주어진 문서에 나타난 한국어 텍스트 문단을 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
이때 텍스트는 그대로 적고, 이미지/도식은 [image]~[/image]라는 구분기호와 적절한 설명으로 대체하여 적어야합니다. 
OCR 결과:'''
            print(query)

            ## Make query
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_path,
                        },
                        {"type": "text", "text": query},
                    ],
                }
            ]

            ## prepare input
            text = text_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = text_tokenizer(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            ## generate
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False, temperature=None)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output = text_tokenizer.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            print(f'Output: {output}')
            #print("# GT alpha:", gt_caption)
            
            ## TODO: calculate accuracy
            pred_caption = output.strip() # alphabet
            ref_texts = [gt_caption]
            hyp_texts = [pred_caption]

            wer_score = wer_metric.compute(references=ref_texts, predictions=hyp_texts)
            cer_score = cer_metric.compute(references=ref_texts, predictions=hyp_texts)

            if wer_score > 1:
                wer_score = 1
            if cer_score > 1:
                cer_score = 1

            print("WER:", (1-wer_score)*100)
            print("CER:", (1-cer_score)*100)

            all_wer_score += (1-wer_score)*100
            all_cer_score += (1-cer_score)*100
            
            rouge_scores = scorer.score(ref_texts[0], hyp_texts[0])
            print("ROUGE:", rouge_scores)
            rouge_2_score = rouge_scores['rouge2'].recall

            all_rouge_score += rouge_2_score*100

    # TODO: make qwen2.5 eval pipeline
    elif category == 'ovis2.5':
        domain_list = os.listdir(dataset_path)
        enable_thinking = thinking_mode # thinking mode
        enable_thinking_budget = True # Only effective if enable_thinking is True.

        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].Dataset_type.strip()
            gt_caption = eval_dataset.loc[i].RAG_Parsing.strip()
            img_path = eval_dataset.loc[i].image.strip()

            ## image load
            img_path = dataset_path + "/" + img_path
            image = Image.open(img_path)
            
            ## prompt
            query = '''당신은 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다. 
다음 주어진 문서에 나타난 한국어 텍스트 문단을 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
이때 텍스트는 그대로 적고, 이미지/도식은 [image]~[/image]라는 구분기호와 적절한 설명으로 대체하여 적어야합니다. 
OCR 결과:'''
            print(query)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query},
                ],
            }]

            ## prepare input
            input_ids, pixel_values, grid_thws = model.preprocess_inputs(
                messages=messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            input_ids = input_ids.cuda()
            pixel_values = pixel_values.cuda() if pixel_values is not None else None
            grid_thws = grid_thws.cuda() if grid_thws is not None else None
            
            # Total tokens for thinking + answer. Ensure: max_new_tokens > thinking_budget + 25
            if enable_thinking:
                max_new_tokens = 4096 # 3072
                thinking_budget = 2048 # 2048
            else:
                max_new_tokens = 4096
                thinking_budget = 4096 # ignore.

            ## generate
            with torch.inference_mode():
                output = model.generate(
                    inputs=input_ids,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    enable_thinking=enable_thinking,
                    enable_thinking_budget=enable_thinking_budget,
                    max_new_tokens=max_new_tokens,
                    thinking_budget=thinking_budget,
                )

                output = model.text_tokenizer.decode(output[0], skip_special_tokens=True)

                if enable_thinking:
                    # <think> ~ </think> answer
                    output = output[output.find('</think>')+len('</think>'):].strip()
                else:
                    pass

            print(f'Output: {output}')

            ## TODO: calculate accuracy
            pred_caption = output.strip() # alphabet
            ref_texts = [gt_caption]
            hyp_texts = [pred_caption]

            wer_score = wer_metric.compute(references=ref_texts, predictions=hyp_texts)
            cer_score = cer_metric.compute(references=ref_texts, predictions=hyp_texts)

            if wer_score > 1:
                wer_score = 1
            if cer_score > 1:
                cer_score = 1

            print("WER:", (1-wer_score)*100)
            print("CER:", (1-cer_score)*100)

            all_wer_score += (1-wer_score)*100
            all_cer_score += (1-cer_score)*100
            
            rouge_scores = scorer.score(ref_texts[0], hyp_texts[0])
            print("ROUGE:", rouge_scores)
            rouge_2_score = rouge_scores['rouge2'].recall

            all_rouge_score += rouge_2_score*100

    # TODO: make qwen3 eval pipeline
    elif category == 'qwen3':
        domain_list = os.listdir(dataset_path)
        #print(eval_dataset.columns)

        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].Dataset_type.strip()
            gt_caption = eval_dataset.loc[i].RAG_Parsing.strip()
            img_path = eval_dataset.loc[i].image.strip()

            ## image load
            img_path = dataset_path + "/" + img_path
            
            ## prompt
            query = '''당신은 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다. 
다음 주어진 문서에 나타난 한국어 텍스트 문단을 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
이때 텍스트는 그대로 적고, 이미지/도식은 [image]~[/image]라는 구분기호와 적절한 설명으로 대체하여 적어야합니다. 
OCR 결과:'''
            print(query)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_path,
                        },
                        {"type": "text", "text": query},
                    ],
                }
            ]

            # Preparation for inference
            inputs = text_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False, temperature=None)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output = text_tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            print(f'Output: {output}')

            ## TODO: calculate accuracy
            pred_caption = output.strip() # alphabet
            ref_texts = [gt_caption]
            hyp_texts = [pred_caption]

            wer_score = wer_metric.compute(references=ref_texts, predictions=hyp_texts)
            cer_score = cer_metric.compute(references=ref_texts, predictions=hyp_texts)

            if wer_score > 1:
                wer_score = 1
            if cer_score > 1:
                cer_score = 1

            print("WER:", (1-wer_score)*100)
            print("CER:", (1-cer_score)*100)

            all_wer_score += (1-wer_score)*100
            all_cer_score += (1-cer_score)*100
            
            rouge_scores = scorer.score(ref_texts[0], hyp_texts[0])
            print("ROUGE:", rouge_scores)
            rouge_2_score = rouge_scores['rouge2'].recall

            all_rouge_score += rouge_2_score*100

    else:
        raise Exception("Not yet implementation")
    
    return all_wer_score/len(eval_dataset), all_cer_score/len(eval_dataset), all_rouge_score/len(eval_dataset)
