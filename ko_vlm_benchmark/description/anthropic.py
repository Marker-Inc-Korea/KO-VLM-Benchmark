import asyncio
import os
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv

from ko_vlm_benchmark.anthropic import claude_multimodal_acomplete


async def generate_description_for_row(
    row_data: dict,
    api_key: str,
    image_base_path: Path,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Generate rich Korean description for a single row using Claude."""
    # Extract data from row
    row_id = row_data["id"]
    doc_type = row_data["doc_type"]
    visual_context = row_data["visual_context"]
    modified_visual_context = row_data["modified_visual_context"]
    original_image_path = row_data["Orig_image"]
    document = row_data["document"]

    # Construct full image path
    full_image_path = str(image_base_path / doc_type / original_image_path)

    # Build prompt
    instruction = (
        "당신은 주어진 이미지를 기반으로, 풍부한 한국어 (korean) description을 생성하는 어시스턴트 입니다.\n"
        "주어진 설명문과 이미지를 기반으로, 정보가 풍부한 한국어 description을 만들어주세요.\n"
        "설명문에 주어진 수치나 표현들을 참고하여 작성하여야 하며, 이미지에 표현된 도식을 설명하는 문장이 포함하도록 만들어야 합니다.\n"
        "생성된 한국어 description은 주어진 설명에 있는 [차트, 도식, 표]와 1대1 매칭을 시켜야하며, 설명문만 출력해주세요."
    )

    query = f"{instruction}\n설명문:\n{modified_visual_context}\n이미지에 대한 풍부한 description:"

    # Generate with semaphore protection
    async with semaphore:
        try:
            response = await claude_multimodal_acomplete(
                api_key=api_key,
                image_path_list=[full_image_path],
                user_text=query,
            )

            if response:
                # Clean unicode bugs if any
                cleaned_response = response.replace("\u0304", "")
                return {
                    "id": row_id,
                    "doc_type": doc_type,
                    "visual_context": visual_context,
                    "modified_visual_context": modified_visual_context,
                    "Orig_image": original_image_path,
                    "document": document,
                    "Anthropic_GT_1": cleaned_response,
                    "status": "success",
                }
            else:
                return {
                    "id": row_id,
                    "doc_type": doc_type,
                    "visual_context": visual_context,
                    "modified_visual_context": modified_visual_context,
                    "Orig_image": original_image_path,
                    "document": document,
                    "Anthropic_GT_1": None,
                    "status": "failed_empty_response",
                }
        except Exception as e:
            click.echo(f"Error processing row {row_id}: {e}")
            return {
                "id": row_id,
                "doc_type": doc_type,
                "visual_context": visual_context,
                "modified_visual_context": modified_visual_context,
                "Orig_image": original_image_path,
                "document": document,
                "Anthropic_GT_1": None,
                "status": f"error: {e!s}",
            }


async def process_batch(
    batch_data: list[dict],
    api_key: str,
    image_base_path: Path,
    max_concurrent: int,
) -> list[dict]:
    """Process a batch of rows concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [generate_description_for_row(row, api_key, image_base_path, semaphore) for row in batch_data]

    results = await asyncio.gather(*tasks)
    return results


def load_checkpoint(output_path: Path) -> tuple[pd.DataFrame | None, int]:
    """Load existing checkpoint if available."""
    if output_path.exists():
        df = pd.read_excel(output_path)
        click.echo(f"Checkpoint found: {len(df)} rows already processed")
        return df, len(df)
    else:
        click.echo("No checkpoint found, starting from scratch")
        return None, 0


def save_checkpoint(results_df: pd.DataFrame, output_path: Path):
    """Save checkpoint to Excel."""
    results_df.to_excel(output_path, index=False)
    click.echo(f"Saved checkpoint: {len(results_df)} rows")


@click.command()
@click.option(
    "--input-excel",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to input Excel file",
)
@click.option(
    "--output-excel",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to output Excel file",
)
@click.option(
    "--image-base-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Base path for images (e.g., ./images/)",
)
@click.option(
    "--batch-size",
    type=int,
    default=10,
    help="Number of rows to process per batch before saving",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=5,
    help="Maximum concurrent API requests",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Resume from checkpoint if available",
)
def main(
    input_excel: Path,
    output_excel: Path,
    image_base_path: Path,
    batch_size: int,
    max_concurrent: int,
    resume: bool,
):
    """Generate rich Korean descriptions using Claude (Anthropic) for multi-page documents."""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    # Load input dataset
    click.echo(f"Loading dataset from {input_excel}")
    dataset = pd.read_excel(input_excel)
    click.echo(f"Total rows in dataset: {len(dataset)}")

    # Load checkpoint if resuming
    checkpoint_df = None
    start_idx = 0
    if resume:
        checkpoint_df, start_idx = load_checkpoint(output_excel)

    # Initialize results dataframe
    if checkpoint_df is not None:
        results_df = checkpoint_df
    else:
        results_df = pd.DataFrame(
            columns=[
                "id",
                "doc_type",
                "visual_context",
                "modified_visual_context",
                "Orig_image",
                "document",
                "Anthropic_GT_1",
            ]
        )

    # Process in batches
    total_rows = len(dataset)
    num_batches = (total_rows - start_idx + batch_size - 1) // batch_size

    click.echo(f"\nStarting processing from row {start_idx}")
    click.echo(f"Batches to process: {num_batches}")
    click.echo(f"Batch size: {batch_size}")
    click.echo(f"Max concurrent requests: {max_concurrent}\n")

    for batch_idx in range(num_batches):
        batch_start = start_idx + (batch_idx * batch_size)
        batch_end = min(batch_start + batch_size, total_rows)

        click.echo(f"{'=' * 60}")
        click.echo(f"Processing batch {batch_idx + 1}/{num_batches}")
        click.echo(f"Rows {batch_start + 1}-{batch_end}/{total_rows}")
        click.echo(f"{'=' * 60}")

        # Prepare batch data
        batch_data = []
        for i in range(batch_start, batch_end):
            row = dataset.iloc[i]
            batch_data.append({
                "id": row.iloc[0],
                "doc_type": str(row.iloc[1]).strip(),
                "visual_context": str(row.iloc[2]).strip(),
                "modified_visual_context": str(row.iloc[3]).strip(),
                "Orig_image": str(row.iloc[-3]).strip(),
                "document": str(row.iloc[-1]).strip(),
            })

        # Process batch
        batch_results = asyncio.run(
            process_batch(
                batch_data=batch_data,
                api_key=api_key,
                image_base_path=image_base_path,
                max_concurrent=max_concurrent,
            )
        )

        # Add results to dataframe
        for result in batch_results:
            # Remove status field before adding to dataframe
            status = result.pop("status", None)
            results_df.loc[len(results_df)] = result
            if status and status != "success":
                click.echo(f"  Row {result['id']}: {status}")

        # Save checkpoint after each batch
        save_checkpoint(results_df, output_excel)

        # Show statistics
        success_count = sum(1 for r in batch_results if r.get("Anthropic_GT_1") is not None)
        click.echo(f"Batch completed: {success_count}/{len(batch_results)} successful\n")

    # Final save
    save_checkpoint(results_df, output_excel)

    click.echo("\n" + "=" * 60)
    click.echo("Processing complete!")
    click.echo(f"Total rows processed: {len(results_df)}")
    click.echo(f"Results saved to: {output_excel}")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
