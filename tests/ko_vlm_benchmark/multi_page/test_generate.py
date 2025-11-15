import pytest
from llama_index.core.llms import MockLLM

from ko_vlm_benchmark.multi_page.generate import generate_two_hop_question

llm = MockLLM()


@pytest.mark.asyncio
async def test_two_hop_incremental():
    new_qa = await generate_two_hop_question(
        desired_answer="Tamaulipas",
        original_document="Nuevo Laredo city is located in the Mexican state of Tamaulipas.",
        added_document="Ciudad Deportiva (Sports City) is a sports complex in Nuevo Laredo, Mexico. It is the home stadium of the Mexican Baseball League team Tecolotes de Nuevo Laredo ...",
        original_query="In which state of Mexico is Nuevo Laredo located?",
        llm=llm,
    )
    assert isinstance(new_qa, str)
