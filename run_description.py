import asyncio
import os
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv

from ko_vlm_benchmark.description.anthropic import generate_description_for_row


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


def load_checkpoint(output_path: Path) -> tuple[pd.DataFrame | None, set]:
    """Load existing checkpoint if available and return indices of failed rows."""
    if output_path.exists():
        df = pd.read_excel(output_path)
        # Find rows where Anthropic_GT_1 is None, empty, or NA (failed rows)
        failed_mask = (
            df["Anthropic_GT_1"].isna()
            | (df["Anthropic_GT_1"] == "")
            | (df["Anthropic_GT_1"].astype(str).str.strip() == "")
        )
        failed_indices = set(df[failed_mask].index.tolist())
        click.echo(f"Checkpoint found: {len(df)} rows already processed")
        if failed_indices:
            click.echo(f"Found {len(failed_indices)} failed rows to retry: {sorted(failed_indices)}")
        return df, failed_indices
    else:
        click.echo("No checkpoint found, starting from scratch")
        return None, set()


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
    failed_indices = set()
    if resume:
        checkpoint_df, failed_indices = load_checkpoint(output_excel)

    # Initialize results dataframe
    if checkpoint_df is not None:
        results_df = checkpoint_df.copy()
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

    # Determine rows to process
    if resume and checkpoint_df is not None:
        # Process only failed rows and new rows not yet in checkpoint
        processed_count = len(checkpoint_df)
        rows_to_process = list(failed_indices) + list(range(processed_count, len(dataset)))
        click.echo(
            f"\nResume mode: Processing {len(failed_indices)} failed rows + {len(dataset) - processed_count} new rows"
        )
    else:
        # Process all rows from scratch
        rows_to_process = list(range(len(dataset)))
        click.echo(f"\nProcessing all {len(rows_to_process)} rows from scratch")

    # Process in batches
    total_rows = len(rows_to_process)
    num_batches = (total_rows + batch_size - 1) // batch_size

    click.echo(f"Batches to process: {num_batches}")
    click.echo(f"Batch size: {batch_size}")
    click.echo(f"Max concurrent requests: {max_concurrent}\n")

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_rows)

        click.echo(f"{'=' * 60}")
        click.echo(f"Processing batch {batch_idx + 1}/{num_batches}")
        click.echo(f"Rows {batch_start + 1}-{batch_end}/{total_rows}")
        click.echo(f"{'=' * 60}")

        # Prepare batch data
        batch_data = []
        batch_result_indices = []
        for i in range(batch_start, batch_end):
            dataset_idx = rows_to_process[i]
            batch_result_indices.append(dataset_idx)
            row = dataset.iloc[dataset_idx]
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

        # Update results in dataframe (replace failed rows or add new rows)
        for idx, result in enumerate(batch_results):
            dataset_idx = batch_result_indices[idx]
            status = result.pop("status", None)

            # Update existing row or add new row
            if dataset_idx < len(results_df):
                # Update existing row (for retry of failed rows)
                for col in result:
                    results_df.at[dataset_idx, col] = result[col]
            else:
                # Add new row
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
