import os
import re
from io import BytesIO

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# from llava.mm_utils import tokenizer_image_token, process_images


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


def VQA_19_Eval(eval_dataset, model, dataset_path, text_tokenizer, visual_tokenizer, category, thinking_mode=False):

    correct_count = 0
    unknown_count = 0
    print(eval_dataset.columns)

    # dataset_path = '/home/kyujin/share/19_VQA_datasets/images/'

    # TODO: make this
    few_shot_prompt = """이미지를 보고 질문에 대한 답변을 제공해주세요. 이때, 반드시 이미지에 제공된 숫자와 단위를 명시해서 답변을 제공해야 합니다.

아래는 이미지에 제시된 숫자 단위가 '백만 원'일 때의 답변 예시입니다.
- 질문: 2017년도 국립청소년산림생태체험센터 건립사업에서 불용된 예산은 얼마인가요?
- 답변: 건립사업에서 불용된 예산은 총 7,131백만 원입니다.

아래는 이미지에 제시된 숫자 단위가 '천 명'일 때의 답변 예시입니다.
- 질문: 2008년 경제활동 인구는 몇 명인가요?
- 답변: 총 24,347천 명입니다.
"""

    # TODO: eval code
    if category == "gemma3":
        domain_list = os.listdir(dataset_path)

        for i in tqdm(range(len(eval_dataset))):
            domain = eval_dataset.loc[i].domain.strip()
            question = eval_dataset.loc[i].question.strip()
            answer = eval_dataset.loc[i].answer.strip()
            key_number = str(eval_dataset.loc[i].key_number)
            img_path = eval_dataset.loc[i].image.strip()

            # gt_number = float(key_number) if '.' in key_number else int(key_number)
            key_number = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", answer)[0]
            # print(answer, key_number)
            unit = answer[answer.find(str(key_number)) + len(str(key_number))]  # 단위

            if "," in key_number:
                key_number = key_number.replace(",", "")
            key_number = float(key_number) if "." in key_number else int(key_number)
            if unit in ["조", "억", "백", "천", "민"]:
                gt_number = str(key_number) + unit
            else:
                gt_number = key_number

            ## image load
            for domain_folder in domain_list:
                if domain in domain_folder:
                    img_path = dataset_path + domain_folder + "/" + img_path
                    break

            # image = Image.open(img_path)
            # image.save('19_vqa.png',"PNG")

            ## Make query
            query = f"{few_shot_prompt}\n\n{question}"
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
                generation = model.generate(**inputs, max_new_tokens=1024, do_sample=False, temperature=None)
                generation = generation[0][input_len:]

            output = text_tokenizer.decode(generation, skip_special_tokens=True)
            print(f"Output: {output}")
            print(f"Answer: {answer}")

            ## TODO: calculate accuracy
            ### 숫자가 한 개면, 바로 비교
            #### 만약, 숫자가 여러개인 경우? -> 아예 같은게 있는지 비교

            ## check answer
            pred_value_list = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", output)

            if len(pred_value_list) == 1:
                pred_value = pred_value_list[0]
                try:
                    # 답변이 숫자만 이루어짐.
                    pred_unit = output[output.find(str(pred_value)) + len(str(pred_value))]  # 단위
                except Exception:
                    pred_unit = ""

                if "," in pred_value:
                    pred_value = pred_value.replace(",", "")

                try:
                    if ("." in pred_value) and (int(float(pred_value)) == float(pred_value)):
                        pred_value = int(float(pred_value))
                    else:
                        pred_value = float(pred_value) if "." in pred_value else int(pred_value)

                except Exception as e:
                    print("Error case:", pred_value, gt_number, e)
                    unknown_count += 1

                if pred_unit in ["조", "억", "백", "천", "민"]:
                    pred_number = str(pred_value) + pred_unit
                else:
                    pred_number = pred_value
                print("GT_number:", gt_number)
                print("PRED_number:", pred_number)

                if pred_number == gt_number:
                    correct_count += 1
                    print("Correct count:", correct_count)

            else:
                for j in range(len(pred_value_list)):
                    pred_value = pred_value_list[j]
                    pred_unit = output[output.find(str(pred_value)) + len(str(pred_value))]  # 단위
                    if "," in pred_value:
                        pred_value = pred_value.replace(",", "")

                    try:
                        if ("." in pred_value) and (int(float(pred_value)) == float(pred_value)):
                            pred_value = int(float(pred_value))
                        else:
                            pred_value = float(pred_value) if "." in pred_value else int(pred_value)

                    except Exception as e:
                        print("Error case:", pred_value, gt_number, e)
                        unknown_count += 1

                    if pred_unit in ["조", "억", "백", "천", "민"]:
                        pred_number = str(pred_value) + pred_unit
                    else:
                        pred_number = pred_value

                    pred_value_list[j] = pred_number

                print("GT_number:", gt_number)
                print("PRED_numbers:", pred_value_list)

                if gt_number in pred_value_list:
                    correct_count += 1
                    print("Correct count:", correct_count)

    elif category == "ovis":
        domain_list = os.listdir(dataset_path)

        for i in tqdm(range(len(eval_dataset))):
            domain = eval_dataset.loc[i].domain.strip()
            question = eval_dataset.loc[i].question.strip()
            answer = eval_dataset.loc[i].answer.strip()
            key_number = str(eval_dataset.loc[i].key_number)
            img_path = eval_dataset.loc[i].image.strip()

            # gt_number = float(key_number) if '.' in key_number else int(key_number)
            key_number = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", answer)[0]
            # print(answer, key_number)
            unit = answer[answer.find(str(key_number)) + len(str(key_number))]  # 단위

            if "," in key_number:
                key_number = key_number.replace(",", "")
            key_number = float(key_number) if "." in key_number else int(key_number)
            if unit in ["조", "억", "백", "천", "민"]:
                gt_number = str(key_number) + unit
            else:
                gt_number = key_number

            ## image load
            for domain_folder in domain_list:
                if domain in domain_folder:
                    img_path = dataset_path + domain_folder + "/" + img_path
                    break

            image = Image.open(img_path)
            # print(img)

            ## Make query
            query = f"<image>\n{few_shot_prompt}\n\n{question}"
            # print(query)

            ## format conversation
            prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
            attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=model.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
            pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

            ## generate output
            with torch.inference_mode():
                gen_kwargs = {
                    "max_new_tokens": 1024,
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
                print(f"Answer: {answer}")

            ## TODO: calculate accuracy
            ### 숫자가 한 개면, 바로 비교
            #### 만약, 숫자가 여러개인 경우? -> 아예 같은게 있는지 비교

            pred_value_list = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", output)

            if len(pred_value_list) == 1:
                pred_value = pred_value_list[0]
                try:
                    # 답변이 숫자만 이루어짐.
                    pred_unit = output[output.find(str(pred_value)) + len(str(pred_value))]  # 단위
                except Exception:
                    pred_unit = ""

                if "," in pred_value:
                    pred_value = pred_value.replace(",", "")

                try:
                    if ("." in pred_value) and (int(float(pred_value)) == float(pred_value)):
                        pred_value = int(float(pred_value))
                    else:
                        pred_value = float(pred_value) if "." in pred_value else int(pred_value)

                except Exception as e:
                    print("Error case:", pred_value, gt_number, e)
                    unknown_count += 1

                if pred_unit in ["조", "억", "백", "천", "민"]:
                    pred_number = str(pred_value) + pred_unit
                else:
                    pred_number = pred_value
                print("GT_number:", gt_number)
                print("PRED_number:", pred_number)

                if pred_number == gt_number:
                    correct_count += 1
                    print("Correct count:", correct_count)

            else:
                for j in range(len(pred_value_list)):
                    pred_value = pred_value_list[j]
                    pred_unit = output[output.find(str(pred_value)) + len(str(pred_value))]  # 단위
                    if "," in pred_value:
                        pred_value = pred_value.replace(",", "")

                    try:
                        if ("." in pred_value) and (int(float(pred_value)) == float(pred_value)):
                            pred_value = int(float(pred_value))
                        else:
                            pred_value = float(pred_value) if "." in pred_value else int(pred_value)

                    except Exception as e:
                        print("Error case:", pred_value, gt_number, e)
                        unknown_count += 1

                    if pred_unit in ["조", "억", "백", "천", "민"]:
                        pred_number = str(pred_value) + pred_unit
                    else:
                        pred_number = pred_value

                    pred_value_list[j] = pred_number

                print("GT_number:", gt_number)
                print("PRED_numbers:", pred_value_list)

                if gt_number in pred_value_list:
                    correct_count += 1
                    print("Correct count:", correct_count)

    elif category == "bllossom":
        domain_list = os.listdir(dataset_path)

        for i in tqdm(range(len(eval_dataset))):
            domain = eval_dataset.loc[i].domain.strip()
            question = eval_dataset.loc[i].question.strip()
            answer = eval_dataset.loc[i].answer.strip()
            key_number = str(eval_dataset.loc[i].key_number)
            img_path = eval_dataset.loc[i].image.strip()

            # gt_number = float(key_number) if '.' in key_number else int(key_number)
            key_number = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", answer)[0]
            # print(answer, key_number)
            unit = answer[answer.find(str(key_number)) + len(str(key_number))]  # 단위

            if "," in key_number:
                key_number = key_number.replace(",", "")
            key_number = float(key_number) if "." in key_number else int(key_number)
            if unit in ["조", "억", "백", "천", "민"]:
                gt_number = str(key_number) + unit
            else:
                gt_number = key_number

            ## image load
            for domain_folder in domain_list:
                if domain in domain_folder:
                    img_path = dataset_path + domain_folder + "/" + img_path
                    break

            image = Image.open(img_path)
            # print(img)

            ## Make query
            query = f"{few_shot_prompt}\n\n{question}"
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
                max_new_tokens=1024,
                temperature=None,
                eos_token_id=text_tokenizer.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                use_cache=True,
            )  # If False, 60 hours
            # print(text_tokenizer.decode(output[0]))

            output = text_tokenizer.decode(output[0])[len(input_text) :].strip()

            print(f"Output:\n{output}")
            print(f"Answer:\n{answer}")

            ## TODO: calculate accuracy
            ### 숫자가 한 개면, 바로 비교
            #### 만약, 숫자가 여러개인 경우? -> 아예 같은게 있는지 비교

            pred_value_list = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", output)

            if len(pred_value_list) == 1:
                pred_value = pred_value_list[0]
                try:
                    # 답변이 숫자만 이루어짐.
                    pred_unit = output[output.find(str(pred_value)) + len(str(pred_value))]  # 단위
                except Exception:
                    pred_unit = ""

                if "," in pred_value:
                    pred_value = pred_value.replace(",", "")

                try:
                    if ("." in pred_value) and (int(float(pred_value)) == float(pred_value)):
                        pred_value = int(float(pred_value))
                    else:
                        pred_value = float(pred_value) if "." in pred_value else int(pred_value)

                except Exception as e:
                    print("Error case:", pred_value, gt_number, e)
                    unknown_count += 1

                if pred_unit in ["조", "억", "백", "천", "민"]:
                    pred_number = str(pred_value) + pred_unit
                else:
                    pred_number = pred_value
                print("GT_number:", gt_number)
                print("PRED_number:", pred_number)

                if pred_number == gt_number:
                    correct_count += 1
                    print("Correct count:", correct_count)

            else:
                for j in range(len(pred_value_list)):
                    pred_value = pred_value_list[j]
                    pred_unit = output[output.find(str(pred_value)) + len(str(pred_value))]  # 단위
                    if "," in pred_value:
                        pred_value = pred_value.replace(",", "")

                    try:
                        if ("." in pred_value) and (int(float(pred_value)) == float(pred_value)):
                            pred_value = int(float(pred_value))
                        else:
                            pred_value = float(pred_value) if "." in pred_value else int(pred_value)

                    except Exception as e:
                        print("Error case:", pred_value, gt_number, e)
                        unknown_count += 1

                    if pred_unit in ["조", "억", "백", "천", "민"]:
                        pred_number = str(pred_value) + pred_unit
                    else:
                        pred_number = pred_value

                    pred_value_list[j] = pred_number

                print("GT_number:", gt_number)
                print("PRED_numbers:", pred_value_list)

                if gt_number in pred_value_list:
                    correct_count += 1
                    print("Correct count:", correct_count)

    elif category == "VARCO-2.0":
        domain_list = os.listdir(dataset_path)

        for i in tqdm(range(len(eval_dataset))):
            domain = eval_dataset.loc[i].domain.strip()
            question = eval_dataset.loc[i].question.strip()
            answer = eval_dataset.loc[i].answer.strip()
            # key_number = str(eval_dataset.loc[i].key_number)
            img_path = eval_dataset.loc[i].image.strip()

            # gt_number = float(key_number) if '.' in key_number else int(key_number)
            key_number = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", answer)[0]
            # print(answer, key_number)
            unit = answer[answer.find(str(key_number)) + len(str(key_number))]  # 단위

            if "," in key_number:
                key_number = key_number.replace(",", "")
            key_number = float(key_number) if "." in key_number else int(key_number)
            if unit in ["조", "억", "백", "천", "민"]:
                gt_number = str(key_number) + unit
            else:
                gt_number = key_number

            ## image load
            for domain_folder in domain_list:
                if domain in domain_folder:
                    img_path = dataset_path + domain_folder + "/" + img_path
                    break

            image = Image.open(img_path)
            # print(img)

            ## Make query
            text = f"{few_shot_prompt}\n\n{question}"
            query = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": text},
                    ],
                },
            ]

            ## preprocessing
            inputs = text_tokenizer.apply_chat_template(
                query, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            ).to(model.device, torch.float16)

            ## generation
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=None,
                    max_new_tokens=1024,
                    use_cache=True,
                )
            generate_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids, strict=True)
            ]
            output = text_tokenizer.decode(generate_ids_trimmed[0], skip_special_tokens=True)

            print(f"Output:\n{output}")

            ## TODO: calculate accuracy
            ### 숫자가 한 개면, 바로 비교
            #### 만약, 숫자가 여러개인 경우? -> 아예 같은게 있는지 비교

            pred_value_list = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", output)

            if len(pred_value_list) == 1:
                pred_value = pred_value_list[0]
                try:
                    # 답변이 숫자만 이루어짐.
                    pred_unit = output[output.find(str(pred_value)) + len(str(pred_value))]  # 단위
                except Exception:
                    pred_unit = ""

                if "," in pred_value:
                    pred_value = pred_value.replace(",", "")

                try:
                    if ("." in pred_value) and (int(float(pred_value)) == float(pred_value)):
                        pred_value = int(float(pred_value))
                    else:
                        pred_value = float(pred_value) if "." in pred_value else int(pred_value)

                except Exception as e:
                    print("Error case:", pred_value, gt_number, e)
                    unknown_count += 1

                if pred_unit in ["조", "억", "백", "천", "민"]:
                    pred_number = str(pred_value) + pred_unit
                else:
                    pred_number = pred_value
                print("GT_number:", gt_number)
                print("PRED_number:", pred_number)

                if pred_number == gt_number:
                    correct_count += 1
                    print("Correct count:", correct_count)

            else:
                for j in range(len(pred_value_list)):
                    pred_value = pred_value_list[j]
                    pred_unit = output[output.find(str(pred_value)) + len(str(pred_value))]  # 단위
                    if "," in pred_value:
                        pred_value = pred_value.replace(",", "")

                    try:
                        if ("." in pred_value) and (int(float(pred_value)) == float(pred_value)):
                            pred_value = int(float(pred_value))
                        else:
                            pred_value = float(pred_value) if "." in pred_value else int(pred_value)

                    except Exception as e:
                        print("Error case:", pred_value, gt_number, e)
                        unknown_count += 1

                    if pred_unit in ["조", "억", "백", "천", "민"]:
                        pred_number = str(pred_value) + pred_unit
                    else:
                        pred_number = pred_value

                    pred_value_list[j] = pred_number

                print("GT_number:", gt_number)
                print("PRED_numbers:", pred_value_list)

                if gt_number in pred_value_list:
                    correct_count += 1
                    print("Correct count:", correct_count)

    elif category == "VARCO":
        domain_list = os.listdir(dataset_path)

        for i in tqdm(range(len(eval_dataset))):
            domain = eval_dataset.loc[i].domain.strip()
            question = eval_dataset.loc[i].question.strip()
            answer = eval_dataset.loc[i].answer.strip()
            # key_number = str(eval_dataset.loc[i].key_number)
            img_path = eval_dataset.loc[i].image.strip()

            # gt_number = float(key_number) if '.' in key_number else int(key_number)
            key_number = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", answer)[0]
            # print(answer, key_number)
            unit = answer[answer.find(str(key_number)) + len(str(key_number))]  # 단위

            if "," in key_number:
                key_number = key_number.replace(",", "")
            key_number = float(key_number) if "." in key_number else int(key_number)
            if unit in ["조", "억", "백", "천", "민"]:
                gt_number = str(key_number) + unit
            else:
                gt_number = key_number

            ## image load
            for domain_folder in domain_list:
                if domain in domain_folder:
                    img_path = dataset_path + domain_folder + "/" + img_path
                    break

            image = Image.open(img_path)
            # print(img)

            ## Make query
            text = f"{few_shot_prompt}\n\n{question}"
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
                    max_new_tokens=1024,
                    use_cache=True,
                )

            output = text_tokenizer.batch_decode(output_ids[0][inputs.input_ids.shape[1] :])
            output = "".join(output).strip()
            if output.endswith(EOS_TOKEN):
                output = output[: -len(EOS_TOKEN)]

            print(f"Output:\n{output}")
            # print(f'Answer:\n{answer}')

            ## TODO: calculate accuracy
            ### 숫자가 한 개면, 바로 비교
            #### 만약, 숫자가 여러개인 경우? -> 아예 같은게 있는지 비교

            pred_value_list = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", output)

            if len(pred_value_list) == 1:
                pred_value = pred_value_list[0]
                try:
                    # 답변이 숫자만 이루어짐.
                    pred_unit = output[output.find(str(pred_value)) + len(str(pred_value))]  # 단위
                except Exception:
                    pred_unit = ""

                if "," in pred_value:
                    pred_value = pred_value.replace(",", "")

                try:
                    if ("." in pred_value) and (int(float(pred_value)) == float(pred_value)):
                        pred_value = int(float(pred_value))
                    else:
                        pred_value = float(pred_value) if "." in pred_value else int(pred_value)

                except Exception as e:
                    print("Error case:", pred_value, gt_number, e)
                    unknown_count += 1

                if pred_unit in ["조", "억", "백", "천", "민"]:
                    pred_number = str(pred_value) + pred_unit
                else:
                    pred_number = pred_value
                print("GT_number:", gt_number)
                print("PRED_number:", pred_number)

                if pred_number == gt_number:
                    correct_count += 1
                    print("Correct count:", correct_count)

            else:
                for j in range(len(pred_value_list)):
                    pred_value = pred_value_list[j]
                    pred_unit = output[output.find(str(pred_value)) + len(str(pred_value))]  # 단위
                    if "," in pred_value:
                        pred_value = pred_value.replace(",", "")

                    try:
                        if ("." in pred_value) and (int(float(pred_value)) == float(pred_value)):
                            pred_value = int(float(pred_value))
                        else:
                            pred_value = float(pred_value) if "." in pred_value else int(pred_value)

                    except Exception as e:
                        print("Error case:", pred_value, gt_number, e)
                        unknown_count += 1

                    if pred_unit in ["조", "억", "백", "천", "민"]:
                        pred_number = str(pred_value) + pred_unit
                    else:
                        pred_number = pred_value

                    pred_value_list[j] = pred_number

                print("GT_number:", gt_number)
                print("PRED_numbers:", pred_value_list)

                if gt_number in pred_value_list:
                    correct_count += 1
                    print("Correct count:", correct_count)

    # TODO: make qwen2.5 eval pipeline
    elif category == "qwen2.5":
        domain_list = os.listdir(dataset_path)

        for i in tqdm(range(len(eval_dataset))):
            domain = eval_dataset.loc[i].domain.strip()
            question = eval_dataset.loc[i].question.strip()
            answer = eval_dataset.loc[i].answer.strip()
            # key_number = str(eval_dataset.loc[i].key_number)
            img_path = eval_dataset.loc[i].image.strip()

            # gt_number = float(key_number) if '.' in key_number else int(key_number)
            key_number = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", answer)[0]
            # print(answer, key_number)
            unit = answer[answer.find(str(key_number)) + len(str(key_number))]  # 단위

            if "," in key_number:
                key_number = key_number.replace(",", "")
            key_number = float(key_number) if "." in key_number else int(key_number)
            if unit in ["조", "억", "백", "천", "민"]:
                gt_number = str(key_number) + unit
            else:
                gt_number = key_number

            ## image load
            for domain_folder in domain_list:
                if domain in domain_folder:
                    img_path = dataset_path + domain_folder + "/" + img_path
                    break

            # image = Image.open(img_path)
            # image.save('19_vqa.png',"PNG") # 꼼수
            # print(img)

            ## Make query
            query = f"{few_shot_prompt}\n\n{question}"
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
                generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, temperature=None)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
                ]
                output = text_tokenizer.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            print(f"Output:\n{output}")

            ## TODO: calculate accuracy
            ### 숫자가 한 개면, 바로 비교
            #### 만약, 숫자가 여러개인 경우? -> 아예 같은게 있는지 비교

            pred_value_list = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", output)

            if len(pred_value_list) == 1:
                pred_value = pred_value_list[0]
                try:
                    # 답변이 숫자만 이루어짐.
                    pred_unit = output[output.find(str(pred_value)) + len(str(pred_value))]  # 단위
                except Exception:
                    pred_unit = ""

                if "," in pred_value:
                    pred_value = pred_value.replace(",", "")

                try:
                    if ("." in pred_value) and (int(float(pred_value)) == float(pred_value)):
                        pred_value = int(float(pred_value))
                    else:
                        pred_value = float(pred_value) if "." in pred_value else int(pred_value)

                except Exception as e:
                    print("Error case:", pred_value, gt_number, e)
                    unknown_count += 1

                pred_number = str(pred_value) + pred_unit if pred_unit in ["조", "억", "백", "천", "민"] else pred_value
                print("GT_number:", gt_number)
                print("PRED_number:", pred_number)

                if pred_number == gt_number:
                    correct_count += 1
                    print("Correct count:", correct_count)

            else:
                for j in range(len(pred_value_list)):
                    pred_value = pred_value_list[j]
                    pred_unit = output[output.find(str(pred_value)) + len(str(pred_value))]  # 단위
                    if "," in pred_value:
                        pred_value = pred_value.replace(",", "")

                    try:
                        if ("." in pred_value) and (int(float(pred_value)) == float(pred_value)):
                            pred_value = int(float(pred_value))
                        else:
                            pred_value = float(pred_value) if "." in pred_value else int(pred_value)

                    except Exception as e:
                        print("Error case:", pred_value, gt_number, e)
                        unknown_count += 1

                    if pred_unit in ["조", "억", "백", "천", "민"]:
                        pred_number = str(pred_value) + pred_unit
                    else:
                        pred_number = pred_value

                    pred_value_list[j] = pred_number

                print("GT_number:", gt_number)
                print("PRED_numbers:", pred_value_list)

                if gt_number in pred_value_list:
                    correct_count += 1
                    print("Correct count:", correct_count)

    elif category == "ovis2.5":
        domain_list = os.listdir(dataset_path)
        enable_thinking = thinking_mode  # thinking mode
        enable_thinking_budget = True  # Only effective if enable_thinking is True.

        for i in tqdm(range(len(eval_dataset))):
            domain = eval_dataset.loc[i].domain.strip()
            question = eval_dataset.loc[i].question.strip()
            answer = eval_dataset.loc[i].answer.strip()
            # key_number = str(eval_dataset.loc[i].key_number)
            img_path = eval_dataset.loc[i].image.strip()

            # gt_number = float(key_number) if '.' in key_number else int(key_number)
            key_number = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", answer)[0]
            # print(answer, key_number)
            unit = answer[answer.find(str(key_number)) + len(str(key_number))]  # 단위

            if "," in key_number:
                key_number = key_number.replace(",", "")
            key_number = float(key_number) if "." in key_number else int(key_number)
            gt_number = str(key_number) + unit if unit in ["조", "억", "백", "천", "민"] else key_number

            ## image load
            for domain_folder in domain_list:
                if domain in domain_folder:
                    img_path = dataset_path + domain_folder + "/" + img_path
                    break

            image = Image.open(img_path)

            ## prompt
            query = f"{few_shot_prompt}\n\n{question}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
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
                max_new_tokens = 2048
                thinking_budget = 1024
            else:
                max_new_tokens = 1024
                thinking_budget = 1024  # ignore.

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
            ### 숫자가 한 개면, 바로 비교
            #### 만약, 숫자가 여러개인 경우? -> 아예 같은게 있는지 비교

            pred_value_list = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", output)

            if len(pred_value_list) == 1:
                pred_value = pred_value_list[0]
                try:
                    # 답변이 숫자만 이루어짐.
                    pred_unit = output[output.find(str(pred_value)) + len(str(pred_value))]  # 단위
                except Exception:
                    pred_unit = ""

                if "," in pred_value:
                    pred_value = pred_value.replace(",", "")

                try:
                    if ("." in pred_value) and (int(float(pred_value)) == float(pred_value)):
                        pred_value = int(float(pred_value))
                    else:
                        pred_value = float(pred_value) if "." in pred_value else int(pred_value)

                except Exception as e:
                    print("Error case:", pred_value, gt_number, e)
                    unknown_count += 1

                pred_number = str(pred_value) + pred_unit if pred_unit in ["조", "억", "백", "천", "민"] else pred_value
                print("GT_number:", gt_number)
                print("PRED_number:", pred_number)

                if pred_number == gt_number:
                    correct_count += 1
                    print("Correct count:", correct_count)

            else:
                for j in range(len(pred_value_list)):
                    pred_value = pred_value_list[j]
                    pred_unit = output[output.find(str(pred_value)) + len(str(pred_value))]  # 단위
                    if "," in pred_value:
                        pred_value = pred_value.replace(",", "")

                    try:
                        if ("." in pred_value) and (int(float(pred_value)) == float(pred_value)):
                            pred_value = int(float(pred_value))
                        else:
                            pred_value = float(pred_value) if "." in pred_value else int(pred_value)

                    except Exception as e:
                        print("Error case:", pred_value, gt_number, e)
                        unknown_count += 1

                    if pred_unit in ["조", "억", "백", "천", "민"]:
                        pred_number = str(pred_value) + pred_unit
                    else:
                        pred_number = pred_value

                    pred_value_list[j] = pred_number

                print("GT_number:", gt_number)
                print("PRED_numbers:", pred_value_list)

                if gt_number in pred_value_list:
                    correct_count += 1
                    print("Correct count:", correct_count)

    # TODO: make qwen3 eval pipeline
    elif category == "qwen3":
        domain_list = os.listdir(dataset_path)

        for i in tqdm(range(len(eval_dataset))):
            domain = eval_dataset.loc[i].domain.strip()
            question = eval_dataset.loc[i].question.strip()
            answer = eval_dataset.loc[i].answer.strip()
            # key_number = str(eval_dataset.loc[i].key_number)
            img_path = eval_dataset.loc[i].image.strip()

            # gt_number = float(key_number) if '.' in key_number else int(key_number)
            key_number = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", answer)[0]
            # print(answer, key_number)
            unit = answer[answer.find(str(key_number)) + len(str(key_number))]  # 단위

            if "," in key_number:
                key_number = key_number.replace(",", "")
            key_number = float(key_number) if "." in key_number else int(key_number)
            gt_number = str(key_number) + unit if unit in ["조", "억", "백", "천", "민"] else key_number

            ## image load
            for domain_folder in domain_list:
                if domain in domain_folder:
                    img_path = dataset_path + domain_folder + "/" + img_path
                    break

            # image = Image.open(img_path)
            # image.save('19_vqa.png',"PNG") # 꼼수
            # print(img)

            ## Make query
            query = f"{few_shot_prompt}\n\n{question}"
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
            ### 숫자가 한 개면, 바로 비교
            #### 만약, 숫자가 여러개인 경우? -> 아예 같은게 있는지 비교

            pred_value_list = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", output)

            if len(pred_value_list) == 1:
                pred_value = pred_value_list[0]
                try:
                    # 답변이 숫자만 이루어짐.
                    pred_unit = output[output.find(str(pred_value)) + len(str(pred_value))]  # 단위
                except Exception:
                    pred_unit = ""

                if "," in pred_value:
                    pred_value = pred_value.replace(",", "")

                try:
                    if ("." in pred_value) and (int(float(pred_value)) == float(pred_value)):
                        pred_value = int(float(pred_value))
                    else:
                        pred_value = float(pred_value) if "." in pred_value else int(pred_value)

                except Exception as e:
                    print("Error case:", pred_value, gt_number, e)
                    unknown_count += 1

                pred_number = str(pred_value) + pred_unit if pred_unit in ["조", "억", "백", "천", "민"] else pred_value
                print("GT_number:", gt_number)
                print("PRED_number:", pred_number)

                if pred_number == gt_number:
                    correct_count += 1
                    print("Correct count:", correct_count)

            else:
                for j in range(len(pred_value_list)):
                    pred_value = pred_value_list[j]
                    pred_unit = output[output.find(str(pred_value)) + len(str(pred_value))]  # 단위
                    if "," in pred_value:
                        pred_value = pred_value.replace(",", "")

                    try:
                        if ("." in pred_value) and (int(float(pred_value)) == float(pred_value)):
                            pred_value = int(float(pred_value))
                        else:
                            pred_value = float(pred_value) if "." in pred_value else int(pred_value)

                    except Exception as e:
                        print("Error case:", pred_value, gt_number, e)
                        unknown_count += 1

                    if pred_unit in ["조", "억", "백", "천", "민"]:
                        pred_number = str(pred_value) + pred_unit
                    else:
                        pred_number = pred_value

                    pred_value_list[j] = pred_number

                print("GT_number:", gt_number)
                print("PRED_numbers:", pred_value_list)

                if gt_number in pred_value_list:
                    correct_count += 1
                    print("Correct count:", correct_count)

    else:
        raise NotImplementedError("Not yet implementation")

    return correct_count / len(eval_dataset)
