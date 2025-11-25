import base64
import aiohttp
import asyncio
import aiofiles

async def encode_image_to_base64(image_path):
    async with aiofiles.open(image_path, 'rb') as f:
        content = await f.read()
    return base64.b64encode(content).decode('utf-8')

async def send_multimodal_request(session, api_key, image_base64, user_text, model="claude-sonnet-4-5"):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "X-Api-Key": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": {
                    "text": user_text,
                    "image_base64": image_base64
                }
            }
        ]
    }

    async with session.post(url, json=data, headers=headers) as resp:
        return await resp.json()

async def claude_multimodal_acomplete(api_key, image_path_list, user_text):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for image_path in image_path_list:
            image_base64 = await encode_image_to_base64(image_path)
            task = asyncio.create_task(send_multimodal_request(session, api_key, image_base64, user_text))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
    return results
