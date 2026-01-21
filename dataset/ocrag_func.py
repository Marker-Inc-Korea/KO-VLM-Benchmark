import torch
import torch.nn.functional as F
import os
import re
import random
import pandas as pd
import re

from tqdm import tqdm
from PIL import Image
from io import BytesIO

from evaluate import load
from rouge_score import rouge_scorer

from qwen_vl_utils import process_vision_info

from sentence_transformers import SentenceTransformer

def check_options(options):

    options_prompt = []
    for i in range(len(options)):
        idx = options.index[i]
        row = options[i]
        # print(idx, row)

        if row is not None:
            prompt = idx + ". " + row.strip()
            options_prompt.append(prompt)

    return options_prompt


def read_image(_bytes):
    try:
        image = Image.open(BytesIO(_bytes))
    except Exception as e:
        raise ValueError(e) from e
    else:
        return image


def OCRAG_Eval(eval_dataset, model, dataset_path, text_tokenizer, visual_tokenizer, category, thinking_mode=False):

    all_wer_score = 0
    all_cer_score = 0
    all_rouge_score = 0
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    print(eval_dataset.columns)

    print("## load metric")
    # WER
    wer_metric = load("wer")
    # CER
    cer_metric = load("cer")

    # TODO: eval code
    if category == "gemma3":
        for i in tqdm(range(len(eval_dataset))):
            gt_caption = eval_dataset.loc[i].RAG_Parsing.strip()
            img_path = eval_dataset.loc[i].image.strip()

            ## image load
            img_path = dataset_path + "/" + img_path
            image = Image.open(img_path)

            ## prompt
            query = """당신은 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다.
다음 주어진 문서에 나타난 한국어 텍스트 문단을 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
이때 텍스트는 그대로 적고, 이미지/도식은 [image]~[/image]라는 구분기호와 적절한 설명으로 대체하여 적어야합니다.
OCR 결과:"""
            print(query)

            # templates
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "image", "url": img_path}, {"type": "text", "text": query}]},
            ]

            ## inference
            inputs = text_tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=True, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda")
            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=4096, do_sample=False, temperature=None)
                generation = generation[0][input_len:]

            output = text_tokenizer.decode(generation, skip_special_tokens=True)
            print(f"Output: {output}")

            ## TODO: calculate accuracy
            pred_caption = output.strip()  # alphabet
            ref_texts = [gt_caption]
            hyp_texts = [pred_caption]

            wer_score = wer_metric.compute(references=ref_texts, predictions=hyp_texts)
            cer_score = cer_metric.compute(references=ref_texts, predictions=hyp_texts)

            if wer_score > 1:
                wer_score = 1
            if cer_score > 1:
                cer_score = 1

            print("WER:", (1 - wer_score) * 100)
            print("CER:", (1 - cer_score) * 100)

            all_wer_score += (1 - wer_score) * 100
            all_cer_score += (1 - cer_score) * 100

            rouge_scores = scorer.score(ref_texts[0], hyp_texts[0])
            print("ROUGE:", rouge_scores)
            rouge_2_score = rouge_scores["rouge2"].recall

            all_rouge_score += rouge_2_score * 100

    elif category == "ovis":
        for i in tqdm(range(len(eval_dataset))):
            gt_caption = eval_dataset.loc[i].RAG_Parsing.strip()
            img_path = eval_dataset.loc[i].image.strip()

            ## image load
            img_path = dataset_path + "/" + img_path
            image = Image.open(img_path)

            ## prompt
            query = """<image>\n당신은 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다.
다음 주어진 문서에 나타난 한국어 텍스트 문단을 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
이때 텍스트는 그대로 적고, 이미지/도식은 [image]~[/image]라는 구분기호와 적절한 설명으로 대체하여 적어야합니다.
OCR 결과:"""
            print(query)

            ## format conversation
            prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
            attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=model.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
            pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

            ## generate output
            with torch.inference_mode():
                gen_kwargs = {
                    "max_new_tokens": 4096,
                    "do_sample": False,
                    "top_p": None,
                    "top_k": None,
                    "temperature": None,
                    "repetition_penalty": None,
                    "eos_token_id": model.generation_config.eos_token_id,
                    "pad_token_id": text_tokenizer.pad_token_id,
                    "use_cache": True,
                }

                output_ids = model.generate(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs
                )[0]
                output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
                print(f"Output: {output}")

            ## TODO: calculate accuracy
            pred_caption = output.strip()  # alphabet
            ref_texts = [gt_caption]
            hyp_texts = [pred_caption]

            wer_score = wer_metric.compute(references=ref_texts, predictions=hyp_texts)
            cer_score = cer_metric.compute(references=ref_texts, predictions=hyp_texts)

            if wer_score > 1:
                wer_score = 1
            if cer_score > 1:
                cer_score = 1

            print("WER:", (1 - wer_score) * 100)
            print("CER:", (1 - cer_score) * 100)

            all_wer_score += (1 - wer_score) * 100
            all_cer_score += (1 - cer_score) * 100

            rouge_scores = scorer.score(ref_texts[0], hyp_texts[0])
            print("ROUGE:", rouge_scores)
            rouge_2_score = rouge_scores["rouge2"].recall

            all_rouge_score += rouge_2_score * 100

    elif category == "bllossom":
        for i in tqdm(range(len(eval_dataset))):
            gt_caption = eval_dataset.loc[i].RAG_Parsing.strip()
            img_path = eval_dataset.loc[i].image.strip()

            ## image load
            img_path = dataset_path + "/" + img_path
            image = Image.open(img_path)

            ## prompt
            query = """당신은 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다.
다음 주어진 문서에 나타난 한국어 텍스트 문단을 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
이때 텍스트는 그대로 적고, 이미지/도식은 [image]~[/image]라는 구분기호와 적절한 설명으로 대체하여 적어야합니다.
OCR 결과:"""
            print(query)

            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]},
            ]

            ## inference
            input_text = text_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = text_tokenizer(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(model.device)

            ## generate output
            output = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=None,
                eos_token_id=text_tokenizer.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                use_cache=True,
            )  # If False, 60 hours
            # print(text_tokenizer.decode(output[0]))

            output = text_tokenizer.decode(output[0])[len(input_text) :].strip()

            print(f"Output: {output}")

            ## TODO: calculate accuracy
            pred_caption = output.strip()  # alphabet
            ref_texts = [gt_caption]
            hyp_texts = [pred_caption]

            wer_score = wer_metric.compute(references=ref_texts, predictions=hyp_texts)
            cer_score = cer_metric.compute(references=ref_texts, predictions=hyp_texts)

            if wer_score > 1:
                wer_score = 1
            if cer_score > 1:
                cer_score = 1

            print("WER:", (1 - wer_score) * 100)
            print("CER:", (1 - cer_score) * 100)

            all_wer_score += (1 - wer_score) * 100
            all_cer_score += (1 - cer_score) * 100

            rouge_scores = scorer.score(ref_texts[0], hyp_texts[0])
            print("ROUGE:", rouge_scores)
            rouge_2_score = rouge_scores["rouge2"].recall

            all_rouge_score += rouge_2_score * 100

    elif category == "VARCO-2.0":
        for i in tqdm(range(len(eval_dataset))):
            gt_caption = eval_dataset.loc[i].RAG_Parsing.strip()
            img_path = eval_dataset.loc[i].image.strip()

            ## image load
            img_path = dataset_path + "/" + img_path
            image = Image.open(img_path)

            ## prompt
            text = """당신은 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다.
다음 주어진 문서에 나타난 한국어 텍스트 문단을 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
이때 텍스트는 그대로 적고, 이미지/도식은 [image]~[/image]라는 구분기호와 적절한 설명으로 대체하여 적어야합니다.
OCR 결과:"""
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
                query, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
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
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids, strict=True)
            ]
            output = text_tokenizer.decode(generate_ids_trimmed[0], skip_special_tokens=True)

            print(f"Output:\n{output}")

            ## TODO: calculate accuracy
            pred_caption = output.strip()  # alphabet
            ref_texts = [gt_caption]
            hyp_texts = [pred_caption]

            wer_score = wer_metric.compute(references=ref_texts, predictions=hyp_texts)
            cer_score = cer_metric.compute(references=ref_texts, predictions=hyp_texts)

            if wer_score > 1:
                wer_score = 1
            if cer_score > 1:
                cer_score = 1

            print("WER:", (1 - wer_score) * 100)
            print("CER:", (1 - cer_score) * 100)

            all_wer_score += (1 - wer_score) * 100
            all_cer_score += (1 - cer_score) * 100

            rouge_scores = scorer.score(ref_texts[0], hyp_texts[0])
            print("ROUGE:", rouge_scores)
            rouge_2_score = rouge_scores["rouge2"].recall

            all_rouge_score += rouge_2_score * 100

    elif category == "VARCO":
        for i in tqdm(range(len(eval_dataset))):
            gt_caption = eval_dataset.loc[i].RAG_Parsing.strip()
            img_path = eval_dataset.loc[i].image.strip()

            ## image load
            img_path = dataset_path + "/" + img_path
            image = Image.open(img_path)

            ## prompt
            text = """당신은 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다.
다음 주어진 문서에 나타난 한국어 텍스트 문단을 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
이때 텍스트는 그대로 적고, 이미지/도식은 [image]~[/image]라는 구분기호와 적절한 설명으로 대체하여 적어야합니다.
OCR 결과:"""
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

            EOS_TOKEN = "<|im_end|>"  # noqa: S105
            inputs = text_tokenizer(images=image, text=prompt, return_tensors="pt").to(torch.float16).to(model.device)

            ## generation
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=None,
                    max_new_tokens=4096,
                    use_cache=True,
                )

            output = text_tokenizer.batch_decode(output_ids[0][inputs.input_ids.shape[1] :])
            output = "".join(output).strip()
            if output.endswith(EOS_TOKEN):
                output = output[: -len(EOS_TOKEN)]

            print(f"Output:\n{output}")

            ## TODO: calculate accuracy
            pred_caption = output.strip()  # alphabet
            ref_texts = [gt_caption]
            hyp_texts = [pred_caption]

            wer_score = wer_metric.compute(references=ref_texts, predictions=hyp_texts)
            cer_score = cer_metric.compute(references=ref_texts, predictions=hyp_texts)

            if wer_score > 1:
                wer_score = 1
            if cer_score > 1:
                cer_score = 1

            print("WER:", (1 - wer_score) * 100)
            print("CER:", (1 - cer_score) * 100)

            all_wer_score += (1 - wer_score) * 100
            all_cer_score += (1 - cer_score) * 100

            rouge_scores = scorer.score(ref_texts[0], hyp_texts[0])
            print("ROUGE:", rouge_scores)
            rouge_2_score = rouge_scores["rouge2"].recall

            all_rouge_score += rouge_2_score * 100

    # TODO: make qwen2.5 eval pipeline
    elif category == "qwen2.5":
        # print(eval_dataset.columns)

        for i in tqdm(range(len(eval_dataset))):
            gt_caption = eval_dataset.loc[i].RAG_Parsing.strip()
            img_path = eval_dataset.loc[i].image.strip()

            ## image load
            img_path = dataset_path + "/" + img_path

            ## prompt
            query = """당신은 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다.
다음 주어진 문서에 나타난 한국어 텍스트 문단을 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
이때 텍스트는 그대로 적고, 이미지/도식은 [image]~[/image]라는 구분기호와 적절한 설명으로 대체하여 적어야합니다.
OCR 결과:"""
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
            text = text_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
                ]
                output = text_tokenizer.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            print(f"Output: {output}")
            # print("# GT alpha:", gt_caption)

            ## TODO: calculate accuracy
            pred_caption = output.strip()  # alphabet
            ref_texts = [gt_caption]
            hyp_texts = [pred_caption]

            wer_score = wer_metric.compute(references=ref_texts, predictions=hyp_texts)
            cer_score = cer_metric.compute(references=ref_texts, predictions=hyp_texts)

            if wer_score > 1:
                wer_score = 1
            if cer_score > 1:
                cer_score = 1

            print("WER:", (1 - wer_score) * 100)
            print("CER:", (1 - cer_score) * 100)

            all_wer_score += (1 - wer_score) * 100
            all_cer_score += (1 - cer_score) * 100

            rouge_scores = scorer.score(ref_texts[0], hyp_texts[0])
            print("ROUGE:", rouge_scores)
            rouge_2_score = rouge_scores["rouge2"].recall

            all_rouge_score += rouge_2_score * 100

    # TODO: make qwen2.5 eval pipeline
    elif category == "ovis2.5":
        enable_thinking = thinking_mode  # thinking mode
        enable_thinking_budget = True  # Only effective if enable_thinking is True.

        for i in tqdm(range(len(eval_dataset))):
            gt_caption = eval_dataset.loc[i].RAG_Parsing.strip()
            img_path = eval_dataset.loc[i].image.strip()

            ## image load
            img_path = dataset_path + "/" + img_path
            image = Image.open(img_path)

            ## prompt
            query = """당신은 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다.
다음 주어진 문서에 나타난 한국어 텍스트 문단을 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
이때 텍스트는 그대로 적고, 이미지/도식은 [image]~[/image]라는 구분기호와 적절한 설명으로 대체하여 적어야합니다.
OCR 결과:"""
            print(query)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": query},
                    ],
                }
            ]

            ## prepare input
            input_ids, pixel_values, grid_thws = model.preprocess_inputs(
                messages=messages, add_generation_prompt=True, enable_thinking=enable_thinking
            )
            input_ids = input_ids.cuda()
            pixel_values = pixel_values.cuda() if pixel_values is not None else None
            grid_thws = grid_thws.cuda() if grid_thws is not None else None

            # Total tokens for thinking + answer. Ensure: max_new_tokens > thinking_budget + 25
            if enable_thinking:
                max_new_tokens = 4096  # 3072
                thinking_budget = 2048  # 2048
            else:
                max_new_tokens = 4096
                thinking_budget = 4096  # ignore.

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
                    output = output[output.find("</think>") + len("</think>") :].strip()
                else:
                    pass

            print(f"Output: {output}")

            ## TODO: calculate accuracy
            pred_caption = output.strip()  # alphabet
            ref_texts = [gt_caption]
            hyp_texts = [pred_caption]

            wer_score = wer_metric.compute(references=ref_texts, predictions=hyp_texts)
            cer_score = cer_metric.compute(references=ref_texts, predictions=hyp_texts)

            if wer_score > 1:
                wer_score = 1
            if cer_score > 1:
                cer_score = 1

            print("WER:", (1 - wer_score) * 100)
            print("CER:", (1 - cer_score) * 100)

            all_wer_score += (1 - wer_score) * 100
            all_cer_score += (1 - cer_score) * 100

            rouge_scores = scorer.score(ref_texts[0], hyp_texts[0])
            print("ROUGE:", rouge_scores)
            rouge_2_score = rouge_scores["rouge2"].recall

            all_rouge_score += rouge_2_score * 100

    # TODO: make qwen3 eval pipeline
    elif category == "qwen3":
        # print(eval_dataset.columns)

        for i in tqdm(range(len(eval_dataset))):
            gt_caption = eval_dataset.loc[i].RAG_Parsing.strip()
            img_path = eval_dataset.loc[i].image.strip()

            ## image load
            img_path = dataset_path + "/" + img_path

            ## prompt
            query = """당신은 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다.
다음 주어진 문서에 나타난 한국어 텍스트 문단을 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
이때 텍스트는 그대로 적고, 이미지/도식은 [image]~[/image]라는 구분기호와 적절한 설명으로 대체하여 적어야합니다.
OCR 결과:"""
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
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )
            inputs = inputs.to(model.device)

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False, temperature=None)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
            ]
            output = text_tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            print(f"Output: {output}")

            ## TODO: calculate accuracy
            pred_caption = output.strip()  # alphabet
            ref_texts = [gt_caption]
            hyp_texts = [pred_caption]

            wer_score = wer_metric.compute(references=ref_texts, predictions=hyp_texts)
            cer_score = cer_metric.compute(references=ref_texts, predictions=hyp_texts)

            if wer_score > 1:
                wer_score = 1
            if cer_score > 1:
                cer_score = 1

            print("WER:", (1 - wer_score) * 100)
            print("CER:", (1 - cer_score) * 100)

            all_wer_score += (1 - wer_score) * 100
            all_cer_score += (1 - cer_score) * 100

            rouge_scores = scorer.score(ref_texts[0], hyp_texts[0])
            print("ROUGE:", rouge_scores)
            rouge_2_score = rouge_scores["rouge2"].recall

            all_rouge_score += rouge_2_score * 100

    else:
        raise NotImplementedError("Not yet implementation")

    return all_wer_score / len(eval_dataset), all_cer_score / len(eval_dataset), all_rouge_score / len(eval_dataset)


# V2 update (kyujin)

def split_region(pred_caption):
    
    pattern = r"(\[image\].*?\[/image\])"
    tokens = re.split(pattern, pred_caption, flags=re.DOTALL)

    pred_text_ocr = ''
    pred_image_ocr = []
    for t in tokens:
        if t.startswith("[image]"):
            #print("IMAGE:", t)
            pred_image_ocr.append(t.strip()[len('[image]'):-len('[/image]')].strip())
        else:
            #print("TEXT:", t.strip())
            pred_text_ocr = pred_text_ocr + t.strip() + '\n'

    return pred_text_ocr.strip(), pred_image_ocr

def OCRAG_Eval_ver2(eval_dataset, 
                model,
                dataset_path,
                text_tokenizer, 
                visual_tokenizer,
                category,
                thinking_mode=False):
    
    all_wer_score = 0
    all_cer_score = 0
    all_rouge_score = 0
    all_sbert_score = 0
    print(eval_dataset.columns)

    print("## load metric")
    # WER
    wer_metric = load("wer")
    # CER
    cer_metric = load("cer")
    # rouge
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    # SBERT
    sbert_model_path = 'jinaai/jina-embeddings-v4'
    sbert_model = SentenceTransformer(sbert_model_path, trust_remote_code=True)
    img_dataset_count = 0

    print('## OCRAG Ver2')

    # TODO: eval code
    if category == 'gemma3':
        
        domain_list = os.listdir(dataset_path)
        
        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].Dataset_type.strip()
            gt_text_ocr = eval_dataset.loc[i].TEXT_OCR.strip()
            img_num = eval_dataset.loc[i].img_num
            img_path = eval_dataset.loc[i].image.strip()
            try:
                gt_image_ocr = eval_dataset.loc[i].IMAGE_DESRIPTION.strip()
                _, gt_image_ocr = split_region(gt_image_ocr)
                
                assert len(gt_image_ocr) == img_num, 'different caption gt?'

                img_dataset_count += 1

            except Exception as E:
                print(E)
                gt_image_ocr = None # there is no image in document
            #print(gt_image_ocr)

            ## image load
            full_img_path = dataset_path + img_path
            #print(full_img_path)
            image = Image.open(full_img_path)
            
            ## prompt
            query = '''당신은 텍스트와 도식/도표에 대해서 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다. 
---
아래의 규칙을 준수하여 답변을 출력해야합니다:
1. 주어진 문서에 나타난 한국어 텍스트 문단과 도표(table)은 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
2. 문서에 나타난 텍스트는 그대로 적고, 도표(table)는 markdown 형식으로 작성하세요.
3. 이미지/도식에 대해서는 적절한 설명문으로 대체하여 작성하세요.
4. 이미지/도식에 대한 설명문은 시작과 끝이 [image], [/image]라는 Tag로 반드시 묶여있어야 합니다.
5. 임의의 문서에 대한 텍스트, 이미지/도식, 도표(table)에 대한 예시입니다. 아래의 예시를 보고 작성 양식을 참고하세요:
A회사 로고 | 인구수 조사

2025년 크리스마스 국내 여행지 인구수 조사

[image] 서울 한복판에 있는 거대한 크리스마스 트리와 앞에 놓여진 여러 선물 장식들 [/image]

이번 2025년 크리스마스에 사람들이 서울을 비롯한 다양한 지역에 몰리는 현상이 발생하였다.
특히나 서울에 2030 청년들이 집중적으로 몰렸으며, 포항에는 노년층들이 여유를 즐기는 모습이 관찰되었다.

[image] 대한민국 지도 위에 각 지역별의 크기를 인구모양으로 나타낸 그림 [/image]

지역별 인구수 표
(단위: 십만)
| 지역 | 인구수 | 혼잡정도 |
| 서울 | 300 | 심함 |
| 경기 | 100 | 약간심함 |
| 대전 | 54 | 보통 |
| 포항 | 3 | 여유 |
---

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
                        {"type": "image", "url": full_img_path},
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
            
            ## calculate metrics
            pred_caption = output.strip()
            pred_text_ocr, pred_image_ocr = split_region(pred_caption)
            #print(pred_text_ocr)
            #print(len(pred_image_ocr))

            ## TODO: Text (WER & CER)
            ref_texts = [gt_text_ocr]
            hyp_texts = [pred_text_ocr]

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

            if gt_image_ocr:

                ## TODO: rouge-score
                #print(gt_image_ocr)
                sub_rouge_score = 0.0
                for k in range(img_num):
                    if k < len(pred_image_ocr):
                        pred_text = pred_image_ocr[k]
                        scores_list = [scorer.score(ref, pred_text) for ref in gt_image_ocr]
                        #print(scores_list)
                        rouge_1_scores_list = torch.tensor([score['rouge1'].recall for score in scores_list])

                        topk_values, topk_indices = torch.topk(rouge_1_scores_list, 1)
                        rouge_score = topk_values[0].item()

                    else:
                        rouge_score = 0.0

                    sub_rouge_score += rouge_score

                sub_rouge_score /= img_num
                print('## All proper rouge-2:', sub_rouge_score*100)
                all_rouge_score += sub_rouge_score*100

                ## TODO: Sentence-BERT similarity
                # Each query must come with a one-sentence instruction that describes the task
                sub_text_similarity = 0.0
                for k in range(img_num):
                    if k < len(pred_image_ocr):
                        texts = [
                            pred_image_ocr[k]
                        ] + gt_image_ocr

                        text_embeddings = sbert_model.encode(sentences=texts, task="text-matching") # (N, 2048)
                        similarity_matrix = torch.tensor(text_embeddings @ text_embeddings.T) # (8,8)
                        #print(similarity_matrix[0])

                        topk_values, topk_indices = torch.topk(similarity_matrix[0], 2)
                        text_similarity = topk_values[1].item()
                        #print('## Proper similarity:', text_similarity*100)

                    else:
                        text_similarity = 0.0

                    sub_text_similarity += text_similarity

                sub_text_similarity = sub_text_similarity/img_num
                print('## All proper similarity:', sub_text_similarity*100)
                all_sbert_score += sub_text_similarity * 100

    elif category == 'ovis':

        domain_list = os.listdir(dataset_path)
        
        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].Dataset_type.strip()
            gt_text_ocr = eval_dataset.loc[i].TEXT_OCR.strip()
            img_num = eval_dataset.loc[i].img_num
            img_path = eval_dataset.loc[i].image.strip()
            try:
                gt_image_ocr = eval_dataset.loc[i].IMAGE_DESRIPTION.strip()
                _, gt_image_ocr = split_region(gt_image_ocr)
                
                assert len(gt_image_ocr) == img_num, 'different caption gt?'

                img_dataset_count += 1

            except Exception as E:
                print(E)
                gt_image_ocr = None # there is no image in document
            #print(gt_image_ocr)

            ## image load
            full_img_path = dataset_path + img_path
            #print(full_img_path)
            image = Image.open(full_img_path)
            #print(image)
            
            ## prompt
            query = '''<image>\n당신은 텍스트와 도식/도표에 대해서 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다. 
---
아래의 규칙을 준수하여 답변을 출력해야합니다:
1. 주어진 문서에 나타난 한국어 텍스트 문단과 도표(table)은 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
2. 문서에 나타난 텍스트는 그대로 적고, 도표(table)는 markdown 형식으로 작성하세요.
3. 이미지/도식에 대해서는 적절한 설명문으로 대체하여 작성하세요.
4. 이미지/도식에 대한 설명문은 시작과 끝이 [image], [/image]라는 Tag로 반드시 묶여있어야 합니다.
5. 임의의 문서에 대한 텍스트, 이미지/도식, 도표(table)에 대한 예시입니다. 아래의 예시를 보고 작성 양식을 참고하세요:
A회사 로고 | 인구수 조사

2025년 크리스마스 국내 여행지 인구수 조사

[image] 서울 한복판에 있는 거대한 크리스마스 트리와 앞에 놓여진 여러 선물 장식들 [/image]

이번 2025년 크리스마스에 사람들이 서울을 비롯한 다양한 지역에 몰리는 현상이 발생하였다.
특히나 서울에 2030 청년들이 집중적으로 몰렸으며, 포항에는 노년층들이 여유를 즐기는 모습이 관찰되었다.

[image] 대한민국 지도 위에 각 지역별의 크기를 인구모양으로 나타낸 그림 [/image]

지역별 인구수 표
(단위: 십만)
| 지역 | 인구수 | 혼잡정도 |
| 서울 | 300 | 심함 |
| 경기 | 100 | 약간심함 |
| 대전 | 54 | 보통 |
| 포항 | 3 | 여유 |
---

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
            
            ## calculate metrics
            pred_caption = output.strip()
            pred_text_ocr, pred_image_ocr = split_region(pred_caption)
            #print(pred_text_ocr)
            #print(len(pred_image_ocr))

            ## TODO: Text (WER & CER)
            ref_texts = [gt_text_ocr]
            hyp_texts = [pred_text_ocr]

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

            if gt_image_ocr:

                ## TODO: rouge-score
                #print(gt_image_ocr)
                sub_rouge_score = 0.0
                for k in range(img_num):
                    if k < len(pred_image_ocr):
                        pred_text = pred_image_ocr[k]
                        scores_list = [scorer.score(ref, pred_text) for ref in gt_image_ocr]
                        #print(scores_list)
                        rouge_1_scores_list = torch.tensor([score['rouge1'].recall for score in scores_list])

                        topk_values, topk_indices = torch.topk(rouge_1_scores_list, 1)
                        rouge_score = topk_values[0].item()

                    else:
                        rouge_score = 0.0

                    sub_rouge_score += rouge_score

                sub_rouge_score /= img_num
                print('## All proper rouge-2:', sub_rouge_score*100)
                all_rouge_score += sub_rouge_score*100

                ## TODO: Sentence-BERT similarity
                # Each query must come with a one-sentence instruction that describes the task
                sub_text_similarity = 0.0
                for k in range(img_num):
                    if k < len(pred_image_ocr):
                        texts = [
                            pred_image_ocr[k]
                        ] + gt_image_ocr

                        text_embeddings = sbert_model.encode(sentences=texts, task="text-matching") # (N, 2048)
                        similarity_matrix = torch.tensor(text_embeddings @ text_embeddings.T) # (8,8)
                        #print(similarity_matrix[0])

                        topk_values, topk_indices = torch.topk(similarity_matrix[0], 2)
                        text_similarity = topk_values[1].item()
                        #print('## Proper similarity:', text_similarity*100)

                    else:
                        text_similarity = 0.0

                    sub_text_similarity += text_similarity

                sub_text_similarity = sub_text_similarity/img_num
                print('## All proper similarity:', sub_text_similarity*100)
                all_sbert_score += sub_text_similarity * 100


    elif category == 'bllossom':
        domain_list = os.listdir(dataset_path)
        
        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].Dataset_type.strip()
            gt_text_ocr = eval_dataset.loc[i].TEXT_OCR.strip()
            img_num = eval_dataset.loc[i].img_num
            img_path = eval_dataset.loc[i].image.strip()
            try:
                gt_image_ocr = eval_dataset.loc[i].IMAGE_DESRIPTION.strip()
                _, gt_image_ocr = split_region(gt_image_ocr)
                
                assert len(gt_image_ocr) == img_num, 'different caption gt?'

                img_dataset_count += 1

            except Exception as E:
                print(E)
                gt_image_ocr = None # there is no image in document
            #print(gt_image_ocr)

            ## image load
            full_img_path = dataset_path + img_path
            #print(full_img_path)
            image = Image.open(full_img_path)
            
            ## prompt
            query = '''당신은 텍스트와 도식/도표에 대해서 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다. 
---
아래의 규칙을 준수하여 답변을 출력해야합니다:
1. 주어진 문서에 나타난 한국어 텍스트 문단과 도표(table)은 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
2. 문서에 나타난 텍스트는 그대로 적고, 도표(table)는 markdown 형식으로 작성하세요.
3. 이미지/도식에 대해서는 적절한 설명문으로 대체하여 작성하세요.
4. 이미지/도식에 대한 설명문은 시작과 끝이 [image], [/image]라는 Tag로 반드시 묶여있어야 합니다.
5. 임의의 문서에 대한 텍스트, 이미지/도식, 도표(table)에 대한 예시입니다. 아래의 예시를 보고 작성 양식을 참고하세요:
A회사 로고 | 인구수 조사

2025년 크리스마스 국내 여행지 인구수 조사

[image] 서울 한복판에 있는 거대한 크리스마스 트리와 앞에 놓여진 여러 선물 장식들 [/image]

이번 2025년 크리스마스에 사람들이 서울을 비롯한 다양한 지역에 몰리는 현상이 발생하였다.
특히나 서울에 2030 청년들이 집중적으로 몰렸으며, 포항에는 노년층들이 여유를 즐기는 모습이 관찰되었다.

[image] 대한민국 지도 위에 각 지역별의 크기를 인구모양으로 나타낸 그림 [/image]

지역별 인구수 표
(단위: 십만)
| 지역 | 인구수 | 혼잡정도 |
| 서울 | 300 | 심함 |
| 경기 | 100 | 약간심함 |
| 대전 | 54 | 보통 |
| 포항 | 3 | 여유 |
---

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
            
            ## calculate metrics
            pred_caption = output.strip()
            pred_text_ocr, pred_image_ocr = split_region(pred_caption)
            #print(pred_text_ocr)
            #print(len(pred_image_ocr))

            ## TODO: Text (WER & CER)
            ref_texts = [gt_text_ocr]
            hyp_texts = [pred_text_ocr]

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

            if gt_image_ocr:

                ## TODO: rouge-score
                #print(gt_image_ocr)
                sub_rouge_score = 0.0
                for k in range(img_num):
                    if k < len(pred_image_ocr):
                        pred_text = pred_image_ocr[k]
                        scores_list = [scorer.score(ref, pred_text) for ref in gt_image_ocr]
                        #print(scores_list)
                        rouge_1_scores_list = torch.tensor([score['rouge1'].recall for score in scores_list])

                        topk_values, topk_indices = torch.topk(rouge_1_scores_list, 1)
                        rouge_score = topk_values[0].item()

                    else:
                        rouge_score = 0.0

                    sub_rouge_score += rouge_score

                sub_rouge_score /= img_num
                print('## All proper rouge-2:', sub_rouge_score*100)
                all_rouge_score += sub_rouge_score*100

                ## TODO: Sentence-BERT similarity
                # Each query must come with a one-sentence instruction that describes the task
                sub_text_similarity = 0.0
                for k in range(img_num):
                    if k < len(pred_image_ocr):
                        texts = [
                            pred_image_ocr[k]
                        ] + gt_image_ocr

                        text_embeddings = sbert_model.encode(sentences=texts, task="text-matching") # (N, 2048)
                        similarity_matrix = torch.tensor(text_embeddings @ text_embeddings.T) # (8,8)
                        #print(similarity_matrix[0])

                        topk_values, topk_indices = torch.topk(similarity_matrix[0], 2)
                        text_similarity = topk_values[1].item()
                        #print('## Proper similarity:', text_similarity*100)

                    else:
                        text_similarity = 0.0

                    sub_text_similarity += text_similarity

                sub_text_similarity = sub_text_similarity/img_num
                print('## All proper similarity:', sub_text_similarity*100)
                all_sbert_score += sub_text_similarity * 100

    elif category == 'VARCO-2.0':
        domain_list = os.listdir(dataset_path)

        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].Dataset_type.strip()
            gt_text_ocr = eval_dataset.loc[i].TEXT_OCR.strip()
            img_num = eval_dataset.loc[i].img_num
            img_path = eval_dataset.loc[i].image.strip()
            try:
                gt_image_ocr = eval_dataset.loc[i].IMAGE_DESRIPTION.strip()
                _, gt_image_ocr = split_region(gt_image_ocr)
                
                assert len(gt_image_ocr) == img_num, 'different caption gt?'

                img_dataset_count += 1

            except Exception as E:
                print(E)
                gt_image_ocr = None # there is no image in document
            #print(gt_image_ocr)

            ## image load
            full_img_path = dataset_path + img_path
            print(full_img_path)
            image = Image.open(full_img_path)
            
            ## prompt
            text = '''당신은 텍스트와 도식/도표에 대해서 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다. 
---
아래의 규칙을 준수하여 답변을 출력해야합니다:
1. 주어진 문서에 나타난 한국어 텍스트 문단과 도표(table)은 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
2. 문서에 나타난 텍스트는 그대로 적고, 도표(table)는 markdown 형식으로 작성하세요.
3. 이미지/도식에 대해서는 적절한 설명문으로 대체하여 작성하세요.
4. 이미지/도식에 대한 설명문은 시작과 끝이 [image], [/image]라는 Tag로 반드시 묶여있어야 합니다.
5. 임의의 문서에 대한 텍스트, 이미지/도식, 도표(table)에 대한 예시입니다. 아래의 예시를 보고 작성 양식을 참고하세요:
A회사 로고 | 인구수 조사

2025년 크리스마스 국내 여행지 인구수 조사

[image] 서울 한복판에 있는 거대한 크리스마스 트리와 앞에 놓여진 여러 선물 장식들 [/image]

이번 2025년 크리스마스에 사람들이 서울을 비롯한 다양한 지역에 몰리는 현상이 발생하였다.
특히나 서울에 2030 청년들이 집중적으로 몰렸으며, 포항에는 노년층들이 여유를 즐기는 모습이 관찰되었다.

[image] 대한민국 지도 위에 각 지역별의 크기를 인구모양으로 나타낸 그림 [/image]

지역별 인구수 표
(단위: 십만)
| 지역 | 인구수 | 혼잡정도 |
| 서울 | 300 | 심함 |
| 경기 | 100 | 약간심함 |
| 대전 | 54 | 보통 |
| 포항 | 3 | 여유 |
---

OCR 결과:'''
            print(text)

            query = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": full_img_path},
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

            ## calculate metrics
            pred_caption = output.strip()
            pred_text_ocr, pred_image_ocr = split_region(pred_caption)
            #print(pred_text_ocr)
            #print(len(pred_image_ocr))

            ## TODO: Text (WER & CER)
            ref_texts = [gt_text_ocr]
            hyp_texts = [pred_text_ocr]

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

            if gt_image_ocr:

                ## TODO: rouge-score
                #print(gt_image_ocr)
                sub_rouge_score = 0.0
                for k in range(img_num):
                    if k < len(pred_image_ocr):
                        pred_text = pred_image_ocr[k]
                        scores_list = [scorer.score(ref, pred_text) for ref in gt_image_ocr]
                        #print(scores_list)
                        rouge_1_scores_list = torch.tensor([score['rouge1'].recall for score in scores_list])

                        topk_values, topk_indices = torch.topk(rouge_1_scores_list, 1)
                        rouge_score = topk_values[0].item()

                    else:
                        rouge_score = 0.0

                    sub_rouge_score += rouge_score

                sub_rouge_score /= img_num
                print('## All proper rouge-2:', sub_rouge_score*100)
                all_rouge_score += sub_rouge_score*100

                ## TODO: Sentence-BERT similarity
                # Each query must come with a one-sentence instruction that describes the task
                sub_text_similarity = 0.0
                for k in range(img_num):
                    if k < len(pred_image_ocr):
                        texts = [
                            pred_image_ocr[k]
                        ] + gt_image_ocr

                        text_embeddings = sbert_model.encode(sentences=texts, task="text-matching") # (N, 2048)
                        similarity_matrix = torch.tensor(text_embeddings @ text_embeddings.T) # (8,8)
                        #print(similarity_matrix[0])

                        topk_values, topk_indices = torch.topk(similarity_matrix[0], 2)
                        text_similarity = topk_values[1].item()
                        #print('## Proper similarity:', text_similarity*100)

                    else:
                        text_similarity = 0.0

                    sub_text_similarity += text_similarity

                sub_text_similarity = sub_text_similarity/img_num
                print('## All proper similarity:', sub_text_similarity*100)
                all_sbert_score += sub_text_similarity * 100
    
    elif category == 'VARCO':
        domain_list = os.listdir(dataset_path)

        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].Dataset_type.strip()
            gt_text_ocr = eval_dataset.loc[i].TEXT_OCR.strip()
            img_num = eval_dataset.loc[i].img_num
            img_path = eval_dataset.loc[i].image.strip()
            try:
                gt_image_ocr = eval_dataset.loc[i].IMAGE_DESRIPTION.strip()
                _, gt_image_ocr = split_region(gt_image_ocr)
                
                assert len(gt_image_ocr) == img_num, 'different caption gt?'

                img_dataset_count += 1

            except Exception as E:
                print(E)
                gt_image_ocr = None # there is no image in document
            #print(gt_image_ocr)

            ## image load
            full_img_path = dataset_path + img_path
            #print(full_img_path)
            image = Image.open(full_img_path)
            
            ## prompt
            text = '''당신은 텍스트와 도식/도표에 대해서 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다. 
---
아래의 규칙을 준수하여 답변을 출력해야합니다:
1. 주어진 문서에 나타난 한국어 텍스트 문단과 도표(table)은 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
2. 문서에 나타난 텍스트는 그대로 적고, 도표(table)는 markdown 형식으로 작성하세요.
3. 이미지/도식에 대해서는 적절한 설명문으로 대체하여 작성하세요.
4. 이미지/도식에 대한 설명문은 시작과 끝이 [image], [/image]라는 Tag로 반드시 묶여있어야 합니다.
5. 임의의 문서에 대한 텍스트, 이미지/도식, 도표(table)에 대한 예시입니다. 아래의 예시를 보고 작성 양식을 참고하세요:
A회사 로고 | 인구수 조사

2025년 크리스마스 국내 여행지 인구수 조사

[image] 서울 한복판에 있는 거대한 크리스마스 트리와 앞에 놓여진 여러 선물 장식들 [/image]

이번 2025년 크리스마스에 사람들이 서울을 비롯한 다양한 지역에 몰리는 현상이 발생하였다.
특히나 서울에 2030 청년들이 집중적으로 몰렸으며, 포항에는 노년층들이 여유를 즐기는 모습이 관찰되었다.

[image] 대한민국 지도 위에 각 지역별의 크기를 인구모양으로 나타낸 그림 [/image]

지역별 인구수 표
(단위: 십만)
| 지역 | 인구수 | 혼잡정도 |
| 서울 | 300 | 심함 |
| 경기 | 100 | 약간심함 |
| 대전 | 54 | 보통 |
| 포항 | 3 | 여유 |
---

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

            ## calculate metrics
            pred_caption = output.strip()
            pred_text_ocr, pred_image_ocr = split_region(pred_caption)
            #print(pred_text_ocr)
            #print(len(pred_image_ocr))

            ## TODO: Text (WER & CER)
            ref_texts = [gt_text_ocr]
            hyp_texts = [pred_text_ocr]

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

            if gt_image_ocr:

                ## TODO: rouge-score
                #print(gt_image_ocr)
                sub_rouge_score = 0.0
                for k in range(img_num):
                    if k < len(pred_image_ocr):
                        pred_text = pred_image_ocr[k]
                        scores_list = [scorer.score(ref, pred_text) for ref in gt_image_ocr]
                        #print(scores_list)
                        rouge_1_scores_list = torch.tensor([score['rouge1'].recall for score in scores_list])

                        topk_values, topk_indices = torch.topk(rouge_1_scores_list, 1)
                        rouge_score = topk_values[0].item()

                    else:
                        rouge_score = 0.0

                    sub_rouge_score += rouge_score

                sub_rouge_score /= img_num
                print('## All proper rouge-1:', sub_rouge_score*100)
                all_rouge_score += sub_rouge_score*100

                ## TODO: Sentence-BERT similarity
                # Each query must come with a one-sentence instruction that describes the task
                sub_text_similarity = 0.0
                for k in range(img_num):
                    if k < len(pred_image_ocr):
                        texts = [
                            pred_image_ocr[k]
                        ] + gt_image_ocr

                        text_embeddings = sbert_model.encode(sentences=texts, task="text-matching") # (N, 2048)
                        similarity_matrix = torch.tensor(text_embeddings @ text_embeddings.T) # (8,8)
                        #print(similarity_matrix[0])

                        topk_values, topk_indices = torch.topk(similarity_matrix[0], 2)
                        text_similarity = topk_values[1].item()
                        #print('## Proper similarity:', text_similarity*100)

                    else:
                        text_similarity = 0.0

                    sub_text_similarity += text_similarity

                sub_text_similarity = sub_text_similarity/img_num
                print('## All proper similarity:', sub_text_similarity*100)
                all_sbert_score += sub_text_similarity * 100
            
    
    # TODO: make qwen2.5 eval pipeline
    elif category == 'qwen2.5':
        domain_list = os.listdir(dataset_path)
        #print(eval_dataset.columns)

        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].Dataset_type.strip()
            gt_text_ocr = eval_dataset.loc[i].TEXT_OCR.strip()
            img_num = eval_dataset.loc[i].img_num
            img_path = eval_dataset.loc[i].image.strip()
            try:
                gt_image_ocr = eval_dataset.loc[i].IMAGE_DESRIPTION.strip()
                _, gt_image_ocr = split_region(gt_image_ocr)
                
                assert len(gt_image_ocr) == img_num, 'different caption gt?'

                img_dataset_count += 1

            except Exception as E:
                print(E)
                gt_image_ocr = None # there is no image in document
            #print(gt_image_ocr)

            ## image load
            full_img_path = dataset_path + img_path
            #print(full_img_path)
            
            ## prompt
            query = '''당신은 텍스트와 도식/도표에 대해서 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다. 
---
아래의 규칙을 준수하여 답변을 출력해야합니다:
1. 주어진 문서에 나타난 한국어 텍스트 문단과 도표(table)은 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
2. 문서에 나타난 텍스트는 그대로 적고, 도표(table)는 markdown 형식으로 작성하세요.
3. 이미지/도식에 대해서는 적절한 설명문으로 대체하여 작성하세요.
4. 이미지/도식에 대한 설명문은 시작과 끝이 [image], [/image]라는 Tag로 반드시 묶여있어야 합니다.
5. 임의의 문서에 대한 텍스트, 이미지/도식, 도표(table)에 대한 예시입니다. 아래의 예시를 보고 작성 양식을 참고하세요:
A회사 로고 | 인구수 조사

2025년 크리스마스 국내 여행지 인구수 조사

[image] 서울 한복판에 있는 거대한 크리스마스 트리와 앞에 놓여진 여러 선물 장식들 [/image]

이번 2025년 크리스마스에 사람들이 서울을 비롯한 다양한 지역에 몰리는 현상이 발생하였다.
특히나 서울에 2030 청년들이 집중적으로 몰렸으며, 포항에는 노년층들이 여유를 즐기는 모습이 관찰되었다.

[image] 대한민국 지도 위에 각 지역별의 크기를 인구모양으로 나타낸 그림 [/image]

지역별 인구수 표
(단위: 십만)
| 지역 | 인구수 | 혼잡정도 |
| 서울 | 300 | 심함 |
| 경기 | 100 | 약간심함 |
| 대전 | 54 | 보통 |
| 포항 | 3 | 여유 |
---

OCR 결과:'''
            print(query)

            ## Make query
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": full_img_path,
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
            
            ## calculate metrics
            pred_caption = output.strip()
            pred_text_ocr, pred_image_ocr = split_region(pred_caption)
            #print(pred_text_ocr)
            #print(len(pred_image_ocr))

            ## TODO: Text (WER & CER)
            ref_texts = [gt_text_ocr]
            hyp_texts = [pred_text_ocr]

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

            if gt_image_ocr:

                ## TODO: rouge-score
                #print(gt_image_ocr)
                sub_rouge_score = 0.0
                for k in range(img_num):
                    if k < len(pred_image_ocr):
                        pred_text = pred_image_ocr[k]
                        scores_list = [scorer.score(ref, pred_text) for ref in gt_image_ocr]
                        #print(scores_list)
                        rouge_1_scores_list = torch.tensor([score['rouge1'].recall for score in scores_list])

                        topk_values, topk_indices = torch.topk(rouge_1_scores_list, 1)
                        rouge_score = topk_values[0].item()

                    else:
                        rouge_score = 0.0

                    sub_rouge_score += rouge_score

                sub_rouge_score /= img_num
                print('## All proper rouge-1:', sub_rouge_score*100)
                all_rouge_score += sub_rouge_score*100

                ## TODO: Sentence-BERT similarity
                # Each query must come with a one-sentence instruction that describes the task
                sub_text_similarity = 0.0
                for k in range(img_num):
                    if k < len(pred_image_ocr):
                        texts = [
                            pred_image_ocr[k]
                        ] + gt_image_ocr

                        text_embeddings = sbert_model.encode(sentences=texts, task="text-matching") # (N, 2048)
                        similarity_matrix = torch.tensor(text_embeddings @ text_embeddings.T) # (8,8)
                        #print(similarity_matrix[0])

                        topk_values, topk_indices = torch.topk(similarity_matrix[0], 2)
                        text_similarity = topk_values[1].item()
                        #print('## Proper similarity:', text_similarity*100)

                    else:
                        text_similarity = 0.0

                    sub_text_similarity += text_similarity

                sub_text_similarity = sub_text_similarity/img_num
                print('## All proper similarity:', sub_text_similarity*100)
                all_sbert_score += sub_text_similarity * 100

    # TODO: make qwen2.5 eval pipeline
    elif category == 'ovis2.5':
        domain_list = os.listdir(dataset_path)
        enable_thinking = thinking_mode # thinking mode
        enable_thinking_budget = True # Only effective if enable_thinking is True.

        for i in tqdm(range(len(eval_dataset))):
            
            domain = eval_dataset.loc[i].Dataset_type.strip()
            gt_text_ocr = eval_dataset.loc[i].TEXT_OCR.strip()
            img_num = eval_dataset.loc[i].img_num
            img_path = eval_dataset.loc[i].image.strip()
            try:
                gt_image_ocr = eval_dataset.loc[i].IMAGE_DESRIPTION.strip()
                _, gt_image_ocr = split_region(gt_image_ocr)
                
                assert len(gt_image_ocr) == img_num, 'different caption gt?'

                img_dataset_count += 1

            except Exception as E:
                print(E)
                gt_image_ocr = None # there is no image in document
            #print(gt_image_ocr)

            ## image load
            full_img_path = dataset_path + img_path
            #print(full_img_path)
            image = Image.open(full_img_path)
            
            ## prompt
            query = '''당신은 텍스트와 도식/도표에 대해서 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다. 
---
아래의 규칙을 준수하여 답변을 출력해야합니다:
1. 주어진 문서에 나타난 한국어 텍스트 문단과 도표(table)은 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
2. 문서에 나타난 텍스트는 그대로 적고, 도표(table)는 markdown 형식으로 작성하세요.
3. 이미지/도식에 대해서는 적절한 설명문으로 대체하여 작성하세요.
4. 이미지/도식에 대한 설명문은 시작과 끝이 [image], [/image]라는 Tag로 반드시 묶여있어야 합니다.
5. 임의의 문서에 대한 텍스트, 이미지/도식, 도표(table)에 대한 예시입니다. 아래의 예시를 보고 작성 양식을 참고하세요:
A회사 로고 | 인구수 조사

2025년 크리스마스 국내 여행지 인구수 조사

[image] 서울 한복판에 있는 거대한 크리스마스 트리와 앞에 놓여진 여러 선물 장식들 [/image]

이번 2025년 크리스마스에 사람들이 서울을 비롯한 다양한 지역에 몰리는 현상이 발생하였다.
특히나 서울에 2030 청년들이 집중적으로 몰렸으며, 포항에는 노년층들이 여유를 즐기는 모습이 관찰되었다.

[image] 대한민국 지도 위에 각 지역별의 크기를 인구모양으로 나타낸 그림 [/image]

지역별 인구수 표
(단위: 십만)
| 지역 | 인구수 | 혼잡정도 |
| 서울 | 300 | 심함 |
| 경기 | 100 | 약간심함 |
| 대전 | 54 | 보통 |
| 포항 | 3 | 여유 |
---

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

            ## calculate metrics
            pred_caption = output.strip()
            pred_text_ocr, pred_image_ocr = split_region(pred_caption)
            #print(pred_text_ocr)
            #print(len(pred_image_ocr))

            ## TODO: Text (WER & CER)
            ref_texts = [gt_text_ocr]
            hyp_texts = [pred_text_ocr]

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

            if gt_image_ocr:

                ## TODO: rouge-score
                #print(gt_image_ocr)
                sub_rouge_score = 0.0
                for k in range(img_num):
                    if k < len(pred_image_ocr):
                        pred_text = pred_image_ocr[k]
                        scores_list = [scorer.score(ref, pred_text) for ref in gt_image_ocr]
                        #print(scores_list)
                        rouge_1_scores_list = torch.tensor([score['rouge1'].recall for score in scores_list])

                        topk_values, topk_indices = torch.topk(rouge_1_scores_list, 1)
                        rouge_score = topk_values[0].item()

                    else:
                        rouge_score = 0.0

                    sub_rouge_score += rouge_score

                sub_rouge_score /= img_num
                print('## All proper rouge-1:', sub_rouge_score*100)
                all_rouge_score += sub_rouge_score*100

                ## TODO: Sentence-BERT similarity
                # Each query must come with a one-sentence instruction that describes the task
                sub_text_similarity = 0.0
                for k in range(img_num):
                    if k < len(pred_image_ocr):
                        texts = [
                            pred_image_ocr[k]
                        ] + gt_image_ocr

                        text_embeddings = sbert_model.encode(sentences=texts, task="text-matching") # (N, 2048)
                        similarity_matrix = torch.tensor(text_embeddings @ text_embeddings.T) # (8,8)
                        #print(similarity_matrix[0])

                        topk_values, topk_indices = torch.topk(similarity_matrix[0], 2)
                        text_similarity = topk_values[1].item()
                        #print('## Proper similarity:', text_similarity*100)

                    else:
                        text_similarity = 0.0

                    sub_text_similarity += text_similarity

                sub_text_similarity = sub_text_similarity/img_num
                print('## All proper similarity:', sub_text_similarity*100)
                all_sbert_score += sub_text_similarity * 100

    # TODO: make qwen3 eval pipeline
    elif category == 'qwen3':
        domain_list = os.listdir(dataset_path)
        #print(eval_dataset.columns)

        for i in tqdm(range(len(eval_dataset))):

            #i = 8
            
            domain = eval_dataset.loc[i].Dataset_type.strip()
            gt_text_ocr = eval_dataset.loc[i].TEXT_OCR.strip()
            img_num = eval_dataset.loc[i].img_num
            img_path = eval_dataset.loc[i].image.strip()
            try:
                gt_image_ocr = eval_dataset.loc[i].IMAGE_DESRIPTION.strip()
                _, gt_image_ocr = split_region(gt_image_ocr)
                
                assert len(gt_image_ocr) == img_num, 'different caption gt?'

                img_dataset_count += 1

            except Exception as E:
                print(E)
                gt_image_ocr = None # there is no image in document
            #print(gt_image_ocr)

            ## image load
            full_img_path = dataset_path + img_path
            #print(full_img_path)
            
            ## prompt
            query = '''당신은 텍스트와 도식/도표에 대해서 Optical Character Recognition (OCR)을 수행하는 AI assistant 입니다. 
---
아래의 규칙을 준수하여 답변을 출력해야합니다:
1. 주어진 문서에 나타난 한국어 텍스트 문단과 도표(table)은 모두 반영하고, 이미지/도식 중 중요하지 않은 내용은 반영하지 않습니다.
2. 문서에 나타난 텍스트는 그대로 적고, 도표(table)는 markdown 형식으로 작성하세요.
3. 이미지/도식에 대해서는 적절한 설명문으로 대체하여 작성하세요.
4. 이미지/도식에 대한 설명문은 시작과 끝이 [image], [/image]라는 Tag로 반드시 묶여있어야 합니다.
5. 임의의 문서에 대한 텍스트, 이미지/도식, 도표(table)에 대한 예시입니다. 아래의 예시를 보고 작성 양식을 참고하세요:
A회사 로고 | 인구수 조사

2025년 크리스마스 국내 여행지 인구수 조사

[image] 서울 한복판에 있는 거대한 크리스마스 트리와 앞에 놓여진 여러 선물 장식들 [/image]

이번 2025년 크리스마스에 사람들이 서울을 비롯한 다양한 지역에 몰리는 현상이 발생하였다.
특히나 서울에 2030 청년들이 집중적으로 몰렸으며, 포항에는 노년층들이 여유를 즐기는 모습이 관찰되었다.

[image] 대한민국 지도 위에 각 지역별의 크기를 인구모양으로 나타낸 그림 [/image]

지역별 인구수 표
(단위: 십만)
| 지역 | 인구수 | 혼잡정도 |
| 서울 | 300 | 심함 |
| 경기 | 100 | 약간심함 |
| 대전 | 54 | 보통 |
| 포항 | 3 | 여유 |
---

OCR 결과:'''
            print(query)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": full_img_path,
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

            ## calculate metrics
            pred_caption = output.strip()
            pred_text_ocr, pred_image_ocr = split_region(pred_caption)
            #print(pred_text_ocr)
            #print(len(pred_image_ocr))

            ## TODO: Text (WER & CER)
            ref_texts = [gt_text_ocr]
            hyp_texts = [pred_text_ocr]

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

            if gt_image_ocr:

                ## TODO: rouge-score
                #print(gt_image_ocr)
                sub_rouge_score = 0.0
                for k in range(img_num):
                    if k < len(pred_image_ocr):
                        pred_text = pred_image_ocr[k]
                        scores_list = [scorer.score(ref, pred_text) for ref in gt_image_ocr]
                        #print(scores_list)
                        rouge_1_scores_list = torch.tensor([score['rouge1'].recall for score in scores_list])

                        topk_values, topk_indices = torch.topk(rouge_1_scores_list, 1)
                        rouge_score = topk_values[0].item()

                    else:
                        rouge_score = 0.0

                    sub_rouge_score += rouge_score

                sub_rouge_score /= img_num
                print('## All proper rouge-1:', sub_rouge_score*100)
                all_rouge_score += sub_rouge_score*100

                ## TODO: Sentence-BERT similarity
                # Each query must come with a one-sentence instruction that describes the task
                sub_text_similarity = 0.0
                for k in range(img_num):
                    if k < len(pred_image_ocr):
                        texts = [
                            pred_image_ocr[k]
                        ] + gt_image_ocr

                        text_embeddings = sbert_model.encode(sentences=texts, task="text-matching") # (N, 2048)
                        similarity_matrix = torch.tensor(text_embeddings @ text_embeddings.T) # (8,8)
                        #print(similarity_matrix[0])

                        topk_values, topk_indices = torch.topk(similarity_matrix[0], 2)
                        text_similarity = topk_values[1].item()
                        #print('## Proper similarity:', text_similarity*100)

                    else:
                        text_similarity = 0.0

                    sub_text_similarity += text_similarity

                sub_text_similarity = sub_text_similarity/img_num
                print('## All proper similarity:', sub_text_similarity*100)
                all_sbert_score += sub_text_similarity * 100

    else:
        raise Exception("Not yet implementation")
    
    print('## img_dataset_count:', img_dataset_count)
    return all_wer_score/len(eval_dataset), all_cer_score/len(eval_dataset), all_rouge_score/img_dataset_count, all_sbert_score/img_dataset_count
