import asyncio
import os
import random
from pathlib import Path
from typing import Any

import click
import pandas as pd
from dotenv import load_dotenv
from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.anthropic import Anthropic
from tqdm import tqdm
from vllm import LLM

from ko_vlm_benchmark.exceptions import MissingColumnError
from ko_vlm_benchmark.multi_page.generate import (
    generate_desired_answer,
    generate_model_answer,
    generate_original_query,
    generate_two_hop_answer,
    generate_two_hop_question,
)
from ko_vlm_benchmark.multi_page.verifier import verify_multipage_question, verify_two_hop_queries


def initialize_models(
    vllm_model_name: str,
    llm_model_name: str,
) -> tuple[LLM, Any, BaseLLM]:
    """Initialize all models at once to avoid redundant overhead."""
    # Initialize vLLM for logprobs verification
    click.echo(f"Initializing vLLM model: {vllm_model_name}")
    vllm_llm = LLM(model=vllm_model_name, tensor_parallel_size=1, gpu_memory_utilization=0.8)
    tokenizer = vllm_llm.get_tokenizer()

    # Initialize LlamaIndex BaseLLM for question generation
    click.echo(f"Initializing BaseLLM: {llm_model_name}")
    base_llm = Anthropic(model=llm_model_name)

    return vllm_llm, tokenizer, base_llm


def load_and_prepare_data(data_path: Path, image_base_path: Path, description_column: str) -> pd.DataFrame:
    """Load label.xlsx and prepare data with image paths grouped by document."""
    click.echo(f"Loading data from {data_path}")
    df = pd.read_excel(data_path)

    # Validate required columns
    required_cols = ["doc_type", description_column, "Orig_image", "document"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise MissingColumnError(missing_cols)

    # Construct full image paths
    df["image_path"] = df.apply(
        lambda row: str(image_base_path / row["doc_type"] / row["Orig_image"]),
        axis=1,
    )

    click.echo(f"Loaded {len(df)} rows from {len(df['document'].unique())} documents")
    return df


def select_page_pairs(
    document_pages: pd.DataFrame,
    num_pairs: int,
) -> list[tuple[int, int]]:
    """Randomly select page pairs from a document."""
    page_indices = list(range(len(document_pages)))

    if len(page_indices) < 2:
        return []

    # Generate all possible pairs
    all_pairs = [(i, j) for i in page_indices for j in page_indices if i < j]

    # Randomly select up to num_pairs
    selected_pairs = random.sample(all_pairs, min(num_pairs, len(all_pairs)))

    return selected_pairs


async def generate_questions_for_document(
    document_id: str,
    document_pages: pd.DataFrame,
    base_llm: BaseLLM,
    num_pairs: int,
    semaphore: asyncio.Semaphore,
    api_key: str,
    description_column: str,
) -> list[dict]:
    """Generate multi-hop questions for a single document."""
    page_pairs = select_page_pairs(document_pages, num_pairs)

    if not page_pairs:
        return []

    tasks = []

    for orig_idx, added_idx in page_pairs:
        orig_page = document_pages.iloc[orig_idx]
        added_page = document_pages.iloc[added_idx]

        async def generate_with_semaphore(
            orig_doc: str,
            added_doc: str,
            orig_img: str,
            added_img: str,
            orig_i: int,
            added_i: int,
        ):
            async with semaphore:
                try:
                    # Generate an original query from the first document
                    original_query = await generate_original_query(
                        document=orig_doc,
                        llm=base_llm,
                    )

                    # Generate the desired answer using MultiModalLLM with the original query
                    desired_answer = await generate_desired_answer(
                        original_query=original_query,
                        document=orig_doc,
                        image_path=orig_img,
                        api_key=api_key,
                    )

                    # Generate the two-hop question
                    question = await generate_two_hop_question(
                        desired_answer=desired_answer,
                        original_document=orig_doc,
                        added_document=added_doc,
                        original_query=original_query,
                        llm=base_llm,
                    )

                    # Generate the two-hop answer using claude-sonnet-4-5
                    two_hop_answer = await generate_two_hop_answer(
                        multi_hop_question=question,
                        documents=[orig_doc, added_doc],
                        llm=base_llm,
                    )
                except Exception as e:
                    click.echo(f"Error generating question for {document_id}: {e}")
                    # Return error record instead of None
                    return {
                        "document_id": document_id,
                        "page_1_idx": orig_i,
                        "page_2_idx": added_i,
                        "page_1_image": orig_img,
                        "page_2_image": added_img,
                        "page_1_context": orig_doc,
                        "page_2_context": added_doc,
                        "generated_question": None,
                        "desired_answer": None,
                        "two_hop_answer": None,
                        "original_query": None,
                        "generation_status": "failed",
                        "generation_error": str(e),
                    }
                else:
                    return {
                        "document_id": document_id,
                        "page_1_idx": orig_i,
                        "page_2_idx": added_i,
                        "page_1_image": orig_img,
                        "page_2_image": added_img,
                        "page_1_context": orig_doc,
                        "page_2_context": added_doc,
                        "generated_question": question,
                        "desired_answer": desired_answer,
                        "two_hop_answer": two_hop_answer,
                        "original_query": original_query,
                        "generation_status": "success",
                        "generation_error": None,
                    }

        task = generate_with_semaphore(
            orig_page[description_column],
            added_page[description_column],
            orig_page["image_path"],
            added_page["image_path"],
            orig_idx,
            added_idx,
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    # Return all results including failed ones
    return results


async def verify_questions_stage1(
    questions: list[dict],
    vllm_llm: LLM,
    tokenizer: Any,
    threshold: float,
    sampling_params: dict,
) -> list[dict]:
    """First stage: Verify using logprobs and information gain."""
    if not questions:
        return []

    click.echo(f"Stage 1 verification: Checking {len(questions)} questions with info gain threshold {threshold}")

    # Separate successful and failed generations
    successful_questions = [q for q in questions if q.get("generation_status") == "success"]
    failed_questions = [q for q in questions if q.get("generation_status") == "failed"]

    # Mark failed generations
    for question in failed_questions:
        question["info_gain"] = None
        question["stage1_passed"] = False

    if not successful_questions:
        click.echo("No successful generations to verify")
        return questions

    queries = [q["generated_question"] for q in successful_questions]
    documents = [(q["page_1_context"], q["page_2_context"]) for q in successful_questions]
    answers = [q["two_hop_answer"] for q in successful_questions]

    info_gains = verify_two_hop_queries(
        queries=queries,
        documents=documents,
        answers=answers,
        llm=vllm_llm,
        tokenizer=tokenizer,
        **sampling_params,
    )

    # Add info_gain and stage1_passed to successful questions
    for question, info_gain in zip(successful_questions, info_gains, strict=True):
        question["info_gain"] = info_gain
        question["stage1_passed"] = info_gain >= threshold

    passed_count = sum(1 for q in questions if q.get("stage1_passed", False))
    click.echo(f"Stage 1 passed: {passed_count}/{len(questions)} questions")

    # Return all questions, not just verified ones
    return questions


async def verify_questions_stage2(
    questions: list[dict],
    semaphore: asyncio.Semaphore,
    vote_count: int,
    api_key: str,
) -> list[dict]:
    """Second stage: Verify multipage requirement using VLM."""
    if not questions:
        return []

    click.echo(f"Stage 2 verification: Checking {len(questions)} questions with vote_count {vote_count}")

    async def verify_single(question: dict):
        # Only verify if stage1 passed
        if not question.get("stage1_passed", False):
            question["stage2_decision"] = "skipped"
            question["stage2_passed"] = False
            return question

        try:
            decision = await verify_multipage_question(
                api_key=api_key,
                query=question["generated_question"],
                image_paths=[question["page_1_image"], question["page_2_image"]],
                semaphore=semaphore,
                vote_count=vote_count,
            )

            question["stage2_decision"] = decision
            question["stage2_passed"] = decision == "yes"
        except Exception as e:
            click.echo(f"Error in stage 2 verification: {e}")
            question["stage2_decision"] = "error"
            question["stage2_passed"] = False
            return question
        else:
            return question

    tasks = [verify_single(q) for q in questions]
    results = await asyncio.gather(*tasks)

    passed_count = sum(1 for r in results if r and r.get("stage2_passed", False))
    click.echo(f"Stage 2 passed: {passed_count}/{len(questions)} questions")

    # Return all questions with verification results
    return results


async def process_batch(
    batch_documents: list[str],
    df: pd.DataFrame,
    vllm_llm: LLM,
    tokenizer: Any,
    base_llm: BaseLLM,
    num_pairs: int,
    threshold: float,
    vote_count: int,
    semaphore_limit: int,
    sampling_params: dict,
    api_key: str,
    description_column: str,
) -> list[dict]:
    """Process a batch of documents."""
    semaphore = asyncio.Semaphore(semaphore_limit)

    all_questions = []

    # Generate questions for all documents in batch
    for doc_id in tqdm(batch_documents, desc="Generating questions"):
        document_pages = df[df["document"] == doc_id].reset_index(drop=True)
        questions = await generate_questions_for_document(
            document_id=doc_id,
            document_pages=document_pages,
            base_llm=base_llm,
            num_pairs=num_pairs,
            semaphore=semaphore,
            api_key=api_key,
            description_column=description_column,
        )
        all_questions.extend(questions)

    click.echo(f"Generated {len(all_questions)} questions for batch")

    # Continue processing even if no questions generated
    if not all_questions:
        click.echo("Warning: No questions were generated for this batch")
        return []

    # Stage 1: Verify with logprobs (returns all questions with stage1 results)
    questions_with_stage1 = await verify_questions_stage1(
        questions=all_questions,
        vllm_llm=vllm_llm,
        tokenizer=tokenizer,
        threshold=threshold,
        sampling_params=sampling_params,
    )

    # Stage 2: Verify multipage requirement (returns all questions with stage2 results)
    questions_with_stage2 = await verify_questions_stage2(
        api_key=api_key,
        questions=questions_with_stage1,
        semaphore=semaphore,
        vote_count=vote_count,
    )

    # Stage 3: Generate model answers for questions that passed stage 2
    stage2_passed_questions = [q for q in questions_with_stage2 if q.get("stage2_passed", False)]
    click.echo(f"Generating model answers for {len(stage2_passed_questions)} questions that passed stage 2")

    async def generate_answer_with_semaphore(question: dict):
        async with semaphore:
            try:
                model_answer = await generate_model_answer(
                    question=question["generated_question"],
                    image_paths=[question["page_1_image"], question["page_2_image"]],
                    api_key=api_key,
                    model="claude-opus-4-5-20251101",
                )
                question["model_answer"] = model_answer
                question["model_answer_status"] = "success"
            except Exception as e:
                click.echo(f"Error generating model answer: {e}")
                question["model_answer"] = None
                question["model_answer_status"] = "failed"
            return question

    # Generate model answers for passed questions
    if stage2_passed_questions:
        tasks = [generate_answer_with_semaphore(q) for q in stage2_passed_questions]
        await asyncio.gather(*tasks)

    # Add model_answer fields to questions that didn't pass stage 2
    for question in questions_with_stage2:
        if not question.get("stage2_passed", False):
            question["model_answer"] = None
            question["model_answer_status"] = "skipped"

    click.echo(
        f"Model answers generated: {sum(1 for q in questions_with_stage2 if q.get('model_answer_status') == 'success')}/{len(stage2_passed_questions)}"
    )

    # Return all questions with their verification and answer results
    return questions_with_stage2


def save_checkpoint(results: list[dict], save_path: Path, batch_idx: int):
    """Save batch results to a single CSV file (append mode)."""
    if not results:
        click.echo(f"No results to save for batch {batch_idx}")
        return

    df_results = pd.DataFrame(results)

    # Add batch index to the dataframe
    df_results["batch_idx"] = batch_idx

    # Single CSV file path
    csv_file = save_path / "results.csv"

    # Append to existing file or create new one
    if csv_file.exists():
        df_results.to_csv(csv_file, mode="a", header=False, index=False)
    else:
        df_results.to_csv(csv_file, mode="w", header=True, index=False)

    # Count statistics
    total = len(results)
    stage1_passed = sum(1 for r in results if r.get("stage1_passed", False))
    stage2_passed = sum(1 for r in results if r.get("stage2_passed", False))

    click.echo(f"Saved {total} results to {csv_file}")
    click.echo(f"  - Stage 1 passed: {stage1_passed}/{total}")
    click.echo(f"  - Stage 2 passed: {stage2_passed}/{total}")


def get_last_checkpoint(save_path: Path) -> int:
    """Find the last completed batch index from the single CSV file."""
    csv_file = save_path / "results.csv"

    if not csv_file.exists():
        return -1

    try:
        df = pd.read_csv(csv_file)
        if "batch_idx" in df.columns and len(df) > 0:
            return int(df["batch_idx"].max())
    except Exception as e:
        click.echo(f"Error reading checkpoint: {e}")
        return -1
    else:
        return -1


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data_multi_page/label_with_description.xlsx"),
    help="Path to label.xlsx file",
)
@click.option(
    "--image-base-path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data_multi_page/images/images"),
    help="Base path for images",
)
@click.option(
    "--save-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to save checkpoint CSV files",
    default=Path("results"),
)
@click.option(
    "--batch-size",
    type=int,
    default=10,
    help="Number of documents to process per batch",
)
@click.option(
    "--num-pairs",
    type=int,
    default=5,
    help="Number of random page pairs to select per document",
)
@click.option(
    "--threshold",
    type=float,
    default=0.0,
    help="Information gain threshold for stage 1 verification",
)
@click.option(
    "--vote-count",
    type=int,
    default=3,
    help="Number of votes for stage 2 verification",
)
@click.option(
    "--semaphore-limit",
    type=int,
    default=16,
    help="Async concurrency limit",
)
@click.option(
    "--vllm-model",
    type=str,
    default="Qwen/Qwen3-8B",
    help="vLLM model name for stage 1 verification",
)
@click.option(
    "--llm-model",
    type=str,
    default="claude-sonnet-4-5-20250929",
    help="LlamaIndex LLM model for question generation",
)
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    help="Temperature for vLLM sampling",
)
@click.option(
    "--max-tokens",
    type=int,
    default=16,
    help="Max tokens for vLLM sampling",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Resume from last checkpoint",
)
@click.option("--subset", default=None, type=int)
@click.option(
    "--description-column",
    type=str,
    default="Anthropic_GT_1",
    help="Column name containing image descriptions",
)
def main(
    data_path: Path,
    image_base_path: Path,
    save_path: Path,
    batch_size: int,
    num_pairs: int,
    threshold: float,
    vote_count: int,
    semaphore_limit: int,
    vllm_model: str,
    llm_model: str,
    temperature: float,
    max_tokens: int,
    resume: bool,
    subset: int | None,
    description_column: str,
):
    """Generate and verify multi-hop questions from multi-page documents."""
    load_dotenv()
    # Create a save directory
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize models
    vllm_llm, tokenizer, base_llm = initialize_models(
        vllm_model_name=vllm_model,
        llm_model_name=llm_model,
    )

    # Prepare sampling params for vLLM
    sampling_params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Load data
    df = load_and_prepare_data(data_path, image_base_path, description_column)
    if subset is not None:
        # Sample documents (not rows) and limit pages per document
        click.echo(f"Applying subset sampling: {subset} documents with max {subset} pages each")

        # Get unique documents and sample them
        all_documents = df["document"].unique()
        sampled_documents = pd.Series(all_documents).sample(n=min(subset, len(all_documents)), random_state=42).tolist()

        # For each sampled document, sample pages if needed
        sampled_dfs = []
        for doc in sampled_documents:
            doc_df = df[df["document"] == doc]
            # If document has more pages than subset, sample them
            if len(doc_df) > subset:
                doc_df = doc_df.sample(n=subset, random_state=42)
            sampled_dfs.append(doc_df)

        df = pd.concat(sampled_dfs, ignore_index=True)
        click.echo(f"Subset: {len(df)} rows from {len(sampled_documents)} documents")

    # Get unique documents
    document_ids = df["document"].unique().tolist()
    click.echo(f"Total documents: {len(document_ids)}")

    # Check for checkpoint
    start_batch = 0
    if resume:
        last_batch = get_last_checkpoint(save_path)
        if last_batch >= 0:
            start_batch = last_batch + 1
            click.echo(f"Resuming from batch {start_batch}")

    # Process in batches
    num_batches = (len(document_ids) + batch_size - 1) // batch_size

    for batch_idx in range(start_batch, num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(document_ids))
        batch_documents = document_ids[batch_start:batch_end]

        click.echo(f"\n{'=' * 60}")
        click.echo(f"Processing batch {batch_idx + 1}/{num_batches}")
        click.echo(f"Documents {batch_start + 1}-{batch_end}/{len(document_ids)}")
        click.echo(f"{'=' * 60}\n")

        # Process batch
        results = asyncio.run(
            process_batch(
                batch_documents=batch_documents,
                df=df,
                vllm_llm=vllm_llm,
                tokenizer=tokenizer,
                base_llm=base_llm,
                num_pairs=num_pairs,
                threshold=threshold,
                vote_count=vote_count,
                semaphore_limit=semaphore_limit,
                sampling_params=sampling_params,
                api_key=os.environ["ANTHROPIC_API_KEY"],
                description_column=description_column,
            )
        )

        # Save checkpoint
        save_checkpoint(results, save_path, batch_idx)

    click.echo("\n" + "=" * 60)
    click.echo("Processing complete!")
    click.echo(f"Results saved to {save_path}")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
