import torch
import os
import re
import random
import pandas as pd

from tqdm import tqdm
from PIL import Image
from io import BytesIO

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


def MMQ_Eval(eval_dataset, 
            model,
            dataset_path,
            text_tokenizer, 
            visual_tokenizer,
            category,
            thinking_mode=False):
    
    type1_correct_count = 0
    type2_correct_count = 0
    print(eval_dataset.columns)

    text_sample = pd.DataFrame(columns=['id', 'type', 'Modified_image_path', 'text_input'])

    # TODO: eval code
    if category == 'gemma3':
        
        domain_list = os.listdir(dataset_path)
        
        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].doc_type.strip()
            Gemini_GT_1 = eval_dataset.loc[i].Gemini_GT_1.strip()
            Gemini_GT_2 = eval_dataset.loc[i].Gemini_GT_2.strip()
            Gemini_GT_3 = eval_dataset.loc[i].Gemini_GT_3.strip()
            Gemini_GT_4 = eval_dataset.loc[i].Gemini_GT_4.strip()
            img_path = eval_dataset.loc[i].Modified_image.strip()

            ## image load
            img_path = dataset_path + domain + "/" + img_path
            image = Image.open(img_path)

            ## shuffle
            alpha_list = ['A', 'B', 'C', 'D']
            index_shuffle = random.sample([1,2,3,4], 4)
            alpha_num = 0
            choices = ''
            gt_alpha = ''
            for index in index_shuffle:
                if index == 1:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_1}\n\n'
                    gt_alpha = alpha_list[alpha_num]

                elif index == 2:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_2}\n\n'

                elif index == 3:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_3}\n\n'

                elif index == 4:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_4}\n\n'

                alpha_num += 1


            ## Make query
            query = f'다음 주어진 [A, B, C, D] 보기 중 이미지에 있는 도식 정보를 가장 잘 설명하는 것을 고르시오. 답변은 무조건 알파벳이 먼저 나와야합니다.\n\n<보기>\n{choices}# 가장 적절한 설명문:'
            #print(query)
            
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
                generation = model.generate(**inputs, max_new_tokens=1024, do_sample=False, temperature=None)
                generation = generation[0][input_len:]

            output = text_tokenizer.decode(generation, skip_special_tokens=True)
            print(f'Output: {output}')
            print("# GT alpha:", gt_alpha)
            
            ## TODO: calculate accuracy
            pred_alpha = output.strip()[0] # alphabet

            if gt_alpha.lower() in pred_alpha.lower():
                if domain == '보고서':
                    type1_correct_count += 1
                elif domain == '보도자료':
                    type2_correct_count += 1
                else: 
                    raise Exception("?????? doc_type:", domain)

                print("Correct count:", type1_correct_count+type2_correct_count)
            
            #break


    elif category == 'ovis':

        domain_list = os.listdir(dataset_path)
        
        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].doc_type.strip()
            Gemini_GT_1 = eval_dataset.loc[i].Gemini_GT_1.strip()
            Gemini_GT_2 = eval_dataset.loc[i].Gemini_GT_2.strip()
            Gemini_GT_3 = eval_dataset.loc[i].Gemini_GT_3.strip()
            Gemini_GT_4 = eval_dataset.loc[i].Gemini_GT_4.strip()
            img_path = eval_dataset.loc[i].Modified_image.strip()

            ## image load
            img_path = dataset_path + domain + "/" + img_path
            image = Image.open(img_path)

            ## shuffle
            alpha_list = ['A', 'B', 'C', 'D']
            index_shuffle = random.sample([1,2,3,4], 4)
            alpha_num = 0
            choices = ''
            gt_alpha = ''
            for index in index_shuffle:
                if index == 1:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_1}\n\n'
                    gt_alpha = alpha_list[alpha_num]

                elif index == 2:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_2}\n\n'

                elif index == 3:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_3}\n\n'

                elif index == 4:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_4}\n\n'

                alpha_num += 1


            ## Make query
            query = f'<image>\n다음 주어진 [A, B, C, D] 보기 중 이미지에 있는 도식 정보를 가장 잘 설명하는 것을 고르시오. 답변은 무조건 알파벳이 먼저 나와야합니다.\n\n<보기>\n{choices}# 가장 적절한 설명문:'
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
                    max_new_tokens=1024,
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
                print("# GT alpha:", gt_alpha)
            
            ## TODO: calculate accuracy
            pred_alpha = output.strip()[0] # alphabet

            if gt_alpha.lower() in pred_alpha.lower():
                if domain == '보고서':
                    type1_correct_count += 1
                elif domain == '보도자료':
                    type2_correct_count += 1
                else: 
                    raise Exception("?????? doc_type:", domain)

                print("Correct count:", type1_correct_count+type2_correct_count)


    elif category == 'bllossom':
        domain_list = os.listdir(dataset_path)
        
        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].doc_type.strip()
            Gemini_GT_1 = eval_dataset.loc[i].Gemini_GT_1.strip()
            Gemini_GT_2 = eval_dataset.loc[i].Gemini_GT_2.strip()
            Gemini_GT_3 = eval_dataset.loc[i].Gemini_GT_3.strip()
            Gemini_GT_4 = eval_dataset.loc[i].Gemini_GT_4.strip()
            img_path = eval_dataset.loc[i].Modified_image.strip()

            ## image load
            img_path = dataset_path + domain + "/" + img_path
            image = Image.open(img_path)

            ## shuffle
            alpha_list = ['A', 'B', 'C', 'D']
            index_shuffle = random.sample([1,2,3,4], 4)
            alpha_num = 0
            choices = ''
            gt_alpha = ''
            for index in index_shuffle:
                if index == 1:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_1}\n\n'
                    gt_alpha = alpha_list[alpha_num]

                elif index == 2:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_2}\n\n'

                elif index == 3:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_3}\n\n'

                elif index == 4:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_4}\n\n'

                alpha_num += 1
            
            ## Make query
            query = f'다음 주어진 [A, B, C, D] 보기 중 이미지에 있는 도식 정보를 가장 잘 설명하는 것을 고르시오. 답변은 무조건 알파벳이 먼저 나와야합니다.\n\n<보기>\n{choices}# 가장 적절한 설명문:'
            #print(query)
            
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
                                    max_new_tokens=1024,
                                    temperature=None,
                                    eos_token_id=text_tokenizer.tokenizer.convert_tokens_to_ids('<|eot_id|>'),
                                    use_cache=True) # If False, 60 hours
            #print(text_tokenizer.decode(output[0]))
            
            output = text_tokenizer.decode(output[0])[len(input_text):].strip()

            print(f'Output: {output}')
            print("# GT alpha:", gt_alpha)
            
            ## TODO: calculate accuracy
            pred_alpha = output.strip()[0] # alphabet

            if gt_alpha.lower() in pred_alpha.lower():
                if domain == '보고서':
                    type1_correct_count += 1
                elif domain == '보도자료':
                    type2_correct_count += 1
                else: 
                    raise Exception("?????? doc_type:", domain)

                print("Correct count:", type1_correct_count+type2_correct_count)
            
            #break

    
    elif category == 'VARCO':
        domain_list = os.listdir(dataset_path)

        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].doc_type.strip()
            Gemini_GT_1 = eval_dataset.loc[i].Gemini_GT_1.strip()
            Gemini_GT_2 = eval_dataset.loc[i].Gemini_GT_2.strip()
            Gemini_GT_3 = eval_dataset.loc[i].Gemini_GT_3.strip()
            Gemini_GT_4 = eval_dataset.loc[i].Gemini_GT_4.strip()
            img_path = eval_dataset.loc[i].Modified_image.strip()

            ## image load
            img_path = dataset_path + domain + "/" + img_path
            image = Image.open(img_path)

            ## shuffle
            alpha_list = ['A', 'B', 'C', 'D']
            index_shuffle = random.sample([1,2,3,4], 4)
            alpha_num = 0
            choices = ''
            gt_alpha = ''
            for index in index_shuffle:
                if index == 1:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_1}\n\n'
                    gt_alpha = alpha_list[alpha_num]

                elif index == 2:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_2}\n\n'

                elif index == 3:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_3}\n\n'

                elif index == 4:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_4}\n\n'

                alpha_num += 1

            ## Make query
            text = f'다음 주어진 [A, B, C, D] 보기 중 이미지에 있는 도식 정보를 가장 잘 설명하는 것을 고르시오. 답변은 무조건 알파벳이 먼저 나와야합니다.\n\n<보기>\n{choices}# 가장 적절한 설명문:'
            #print(text)

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
                    max_new_tokens=1024,
                    use_cache=True,
                )

            output = text_tokenizer.batch_decode(output_ids[0][inputs.input_ids.shape[1]:])
            output = ''.join(output).strip()
            if output.endswith(EOS_TOKEN):
                output = output[: -len(EOS_TOKEN)]

            print(f'Output:\n{output}')
            print("# GT alpha:", gt_alpha)

            ## TODO: calculate accuracy
            pred_alpha = output.strip()[0] # alphabet

            if gt_alpha.lower() in pred_alpha.lower():
                if domain == '보고서':
                    type1_correct_count += 1
                elif domain == '보도자료':
                    type2_correct_count += 1
                else: 
                    raise Exception("?????? doc_type:", domain)

                print("Correct count:", type1_correct_count+type2_correct_count)
            
            #break
            
    
    # TODO: make qwen2.5 eval pipeline
    elif category == 'qwen2.5':
        domain_list = os.listdir(dataset_path)

        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].doc_type.strip()
            Gemini_GT_1 = eval_dataset.loc[i].Gemini_GT_1.strip()
            Gemini_GT_2 = eval_dataset.loc[i].Gemini_GT_2.strip()
            Gemini_GT_3 = eval_dataset.loc[i].Gemini_GT_3.strip()
            Gemini_GT_4 = eval_dataset.loc[i].Gemini_GT_4.strip()
            img_path = eval_dataset.loc[i].Modified_image.strip()

            ## image load
            img_path = dataset_path + domain + "/" + img_path

            ## shuffle
            alpha_list = ['A', 'B', 'C', 'D']
            #alpha_list = ['ⓐ', 'ⓑ', 'ⓒ', 'ⓓ']
            index_shuffle = random.sample([1,2,3,4], 4)
            #index_shuffle = [1,2,3,4]
            alpha_num = 0
            choices = ''
            gt_alpha = ''
            for index in index_shuffle:
                if index == 1:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_1}\n\n'
                    gt_alpha = alpha_list[alpha_num]

                elif index == 2:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_2}\n\n'

                elif index == 3:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_3}\n\n'

                elif index == 4:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_4}\n\n'

                alpha_num += 1
            

            ## prompt
            query = f'다음 주어진 [A, B, C, D] 보기 중 이미지에 있는 도식 정보를 가장 잘 설명하는 것을 고르시오. 답변은 무조건 알파벳이 먼저 나와야합니다.\n\n<보기>\n{choices}# 가장 적절한 설명문:'
            #query = f'다음 주어진 [ⓐ, ⓑ, ⓒ, ⓓ] 선택지 중 이미지에 있는 도식 정보를 가장 잘 설명하는 것을 고르시오. 답변은 무조건 알파벳 기호가 먼저 나와야합니다.\n\n{choices}\n가장 적절한 설명문:'
            #print(query)

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
                generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, temperature=None)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output = text_tokenizer.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            print(f'Output: {output}')
            print("# GT alpha:", gt_alpha)

            ## TODO: calculate accuracy
            pred_alpha = output.strip()[0] # alphabet

            if gt_alpha.lower() in pred_alpha.lower():
                if domain == '보고서':
                    type1_correct_count += 1
                elif domain == '보도자료':
                    type2_correct_count += 1
                else: 
                    raise Exception("?????? doc_type:", domain)

                print("Correct count:", type1_correct_count+type2_correct_count)
            
            #break

    # TODO: make qwen2.5 eval pipeline
    elif category == 'ovis2.5':
        domain_list = os.listdir(dataset_path)
        enable_thinking = thinking_mode # thinking mode
        enable_thinking_budget = True # Only effective if enable_thinking is True.

        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].doc_type.strip()
            Gemini_GT_1 = eval_dataset.loc[i].Gemini_GT_1.strip()
            Gemini_GT_2 = eval_dataset.loc[i].Gemini_GT_2.strip()
            Gemini_GT_3 = eval_dataset.loc[i].Gemini_GT_3.strip()
            Gemini_GT_4 = eval_dataset.loc[i].Gemini_GT_4.strip()
            img_path = eval_dataset.loc[i].Modified_image.strip()

            ## image load
            img_path = dataset_path + domain + "/" + img_path
            image = Image.open(img_path)

            ## shuffle
            alpha_list = ['A', 'B', 'C', 'D']
            index_shuffle = random.sample([1,2,3,4], 4)
            alpha_num = 0
            choices = ''
            gt_alpha = ''
            for index in index_shuffle:
                if index == 1:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_1}\n\n'
                    gt_alpha = alpha_list[alpha_num]

                elif index == 2:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_2}\n\n'

                elif index == 3:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_3}\n\n'

                elif index == 4:
                    choices += f'{alpha_list[alpha_num]}. {Gemini_GT_4}\n\n'

                alpha_num += 1
            

            ## prompt
            query = f'다음 주어진 [A, B, C, D] 보기 중 이미지에 있는 도식 정보를 가장 잘 설명하는 것을 고르시오. 답변은 무조건 알파벳이 먼저 나와야합니다.\n\n<보기>\n{choices}# 가장 적절한 설명문:'
            #print(query)

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
                max_new_tokens = 3072 # 3072
                thinking_budget = 2048 # 2048
            else:
                max_new_tokens = 1024
                thinking_budget = 1024 # ignore.

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
            print("# GT alpha:", gt_alpha)

            ## TODO: calculate accuracy
            pred_alpha = output.strip()[0] # alphabet

            if gt_alpha.lower() in pred_alpha.lower():
                if domain == '보고서':
                    type1_correct_count += 1
                elif domain == '보도자료':
                    type2_correct_count += 1
                else: 
                    raise Exception("?????? doc_type:", domain)

                print("Correct count:", type1_correct_count+type2_correct_count)
            
            #break

    else:
        raise Exception("Not yet implementation")
    
    return (type1_correct_count+type2_correct_count)/len(eval_dataset), type1_correct_count/100, type2_correct_count/100
