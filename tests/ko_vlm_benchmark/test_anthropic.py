import os
import pathlib
from dotenv import load_dotenv
import pytest
from ko_vlm_benchmark.anthropic import claude_multimodal_acomplete


root_dir = pathlib.PurePath(os.path.dirname(os.path.abspath(__file__))).parent

@pytest.mark.api
@pytest.mark.asyncio
async def test_claude_multimodal_acomplete():
    load_dotenv()
    api_key = os.environ["ANTHROPIC_API_KEY"]
    sample_image_paths = [
        os.path.join(root_dir, "resources", "baseball-die-profile.webp"),
        os.path.join(root_dir, "resources", "lotte.jpeg"),
    ]

    user_text = "Describe the content of the images."
    results = await claude_multimodal_acomplete(api_key, sample_image_paths, user_text)
    assert results is not None
