import asyncio
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from ko_vlm_benchmark.description.anthropic import generate_description_for_row


@pytest.fixture(scope="module")
def setup_env():
    """Load environment variables once for all tests."""
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not found in environment variables")
    return api_key


@pytest.mark.api
@pytest.mark.asyncio
async def test_generate_description_for_row(setup_env):
    """Test that generate_description_for_row works without errors."""
    api_key = setup_env

    # Use test resources directory
    test_resources = Path(__file__).parent.parent.parent / "resources"

    # Sample test data using actual test image
    row_data = {
        "id": "test_001",
        "doc_type": "",  # Empty since image is directly in resources folder
        "visual_context": "Sample visual context",
        "modified_visual_context": "This is a score board between KIA and LOTTE.",
        "Orig_image": "lotte.jpeg",
        "document": "test_document",
    }

    # Use test resources as image base path
    image_base_path = test_resources

    # Skip test if image doesn't exist
    test_image_path = image_base_path / row_data["doc_type"] / row_data["Orig_image"]
    if not test_image_path.exists():
        pytest.skip(f"Test image {test_image_path} does not exist")

    # Create semaphore
    semaphore = asyncio.Semaphore(1)

    # Run the function
    result = await generate_description_for_row(
        row_data=row_data,
        api_key=api_key,
        image_base_path=image_base_path,
        semaphore=semaphore,
    )

    # Basic assertions - just check that it runs without error
    assert result is not None
    assert "id" in result
    assert result["id"] == "test_001"
    assert "Anthropic_GT_1" in result
    assert "status" in result
