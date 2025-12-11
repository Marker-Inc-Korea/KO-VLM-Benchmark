"""Batch processing script for Nano Banana Pipeline.

This script processes large datasets through the pipeline with four commands:

Batch mode (for large datasets):
1. `process` - Run Steps 1-4 for all items, generate JSONL for batch API
2. `submit` - Submit the batch job to Gemini API
3. `retrieve` - Check batch status and retrieve/save results

Full mode (for smaller datasets):
4. `run-all` - Run Steps 1-5 for all items in one go

Usage:
    # Batch mode (3-step process)
    python run_nano_banana_batch.py process --input-excel data.xlsx --output-dir output/
    python run_nano_banana_batch.py submit --output-dir output/
    python run_nano_banana_batch.py retrieve --output-dir output/

    # Full mode (single command)
    python run_nano_banana_batch.py run-all --input-excel data.xlsx --output-dir output/
"""

import asyncio
import json
import logging
import time
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from ko_vlm_benchmark.nano_banana_pipeline import (
    NanoBananaPipeline,
    PartialPipelineOutput,
    PipelineConfig,
    PipelineInput,
    PipelineOutput,
)
from ko_vlm_benchmark.nano_banana_pipeline.chains import build_batch_request

# Type alias for pipeline outputs
PipelineResult = PartialPipelineOutput | PipelineOutput

logger = logging.getLogger("KoVLMBenchmark")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


async def process_dataset(
    df: pd.DataFrame,
    pipeline: NanoBananaPipeline,
    image_path_col: str,
    visual_desc_col: str,
    max_concurrent: int = 8,
) -> list[tuple[int, PipelineResult | None, str | None]]:
    """Process all dataset items through the pipeline.

    Works with both partial (Steps 1-4) and full (Steps 1-5) pipeline modes.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_row(idx: int, row: pd.Series) -> tuple[int, PipelineResult | None, str | None]:
        async with semaphore:
            try:
                input_data = PipelineInput(
                    image_path=str(row[image_path_col]),
                    visual_description=str(row[visual_desc_col]),
                )
                result = await pipeline.arun(input_data)
            except Exception as e:
                return (idx, None, str(e))
            else:
                return (idx, result, None)
            finally:
                logger.info(f"Processed row {idx}")

    tasks = [process_row(int(idx), row) for idx, row in df.iterrows()]
    return list(await asyncio.gather(*tasks))


def create_batch_jsonl(
    client: genai.Client,
    partial_outputs: list[tuple[int, PartialPipelineOutput | None, str | None]],
    output_path: Path,
) -> int:
    """Create JSONL file for Gemini batch API using shared build_batch_request."""
    count = 0
    with open(output_path, "w") as f:
        for idx, partial_output, _error in partial_outputs:
            if partial_output is None:
                continue

            # Use shared function from image_generation.py
            request = build_batch_request(
                client,
                original_image_path=partial_output["original_image_path"],
                image_prompt=partial_output["image_generation_prompt"],
            )
            line = json.dumps({"key": f"request-{idx}", "request": request})
            f.write(line + "\n")
            count += 1

    return count


@click.group()
def cli() -> None:
    """Nano Banana Pipeline batch processing commands."""
    pass


@cli.command()
@click.option(
    "--input-excel",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input Excel file with dataset",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("output/nano_banana_batch"),
    help="Output directory for results",
)
@click.option("--image-path-col", type=str, default="img_path")
@click.option("--visual-desc-col", type=str, default="Anthropic_GT_1")
@click.option("--max-concurrent", type=int, default=5)
@click.option("--start-idx", type=int, default=None, help="Start index for subset processing (inclusive)")
@click.option("--end-idx", type=int, default=None, help="End index for subset processing (exclusive)")
def process(
    input_excel: Path,
    output_dir: Path,
    image_path_col: str,
    visual_desc_col: str,
    max_concurrent: int,
    start_idx: int | None,
    end_idx: int | None,
) -> None:
    """Run Steps 1-4 and generate JSONL for batch API."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    click.echo(f"Loading dataset from {input_excel}...")
    df = pd.read_excel(input_excel)
    total_rows = len(df)
    click.echo(f"Loaded {total_rows} rows")

    # Apply subset filtering
    if start_idx is not None or end_idx is not None:
        start = start_idx if start_idx is not None else 0
        end = end_idx if end_idx is not None else total_rows
        df = df.iloc[start:end].reset_index(drop=True)
        click.echo(f"Processing subset: rows {start} to {end} ({len(df)} rows)")

    # Initialize pipeline with skip_image_generation=True
    config = PipelineConfig(skip_image_generation=True)
    pipeline = NanoBananaPipeline(config)

    # Process all items through Steps 1-4
    click.echo("Running Steps 1-4...")
    partial_outputs = asyncio.run(
        process_dataset(
            df=df,
            pipeline=pipeline,
            image_path_col=image_path_col,
            visual_desc_col=visual_desc_col,
            max_concurrent=max_concurrent,
        )
    )

    successes = sum(1 for _, output, _ in partial_outputs if output is not None)
    failures = sum(1 for _, _, error in partial_outputs if error is not None)
    click.echo(f"Steps 1-4 complete: {successes} successes, {failures} failures")

    # Save partial results
    partial_results_path = output_dir / "partial_results.json"
    with open(partial_results_path, "w") as f:
        serializable = []
        for idx, output, error in partial_outputs:
            serializable.append({
                "index": idx,
                "output": dict(output) if output else None,
                "error": error,
            })
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    click.echo(f"Saved partial results to {partial_results_path}")

    # Create JSONL for batch API
    jsonl_path = output_dir / "batch_requests.jsonl"
    click.echo("Creating batch JSONL...")
    google_client = genai.Client(api_key=config.google_api_key)
    num_requests = create_batch_jsonl(google_client, partial_outputs, jsonl_path)
    click.echo(f"Created {num_requests} batch requests at {jsonl_path}")

    # Save config info
    config_path = output_dir / "batch_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "gemini_model": config.gemini_image_model,
                "num_requests": num_requests,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )

    click.echo(f"\nNext step: python run_nano_banana_batch.py submit --output-dir {output_dir}")


@cli.command()
@click.option(
    "--output-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Output directory with batch_requests.jsonl",
)
def submit(output_dir: Path) -> None:
    """Submit batch job to Gemini API."""
    jsonl_path = output_dir / "batch_requests.jsonl"
    config_path = output_dir / "batch_config.json"

    if not jsonl_path.exists():
        click.echo(f"Error: {jsonl_path} not found. Run 'process' first.")
        return

    # Load config
    with open(config_path) as f:
        batch_config = json.load(f)

    config = PipelineConfig()
    client = genai.Client(api_key=config.google_api_key)

    click.echo(f"Uploading {jsonl_path}...")
    uploaded_file = client.files.upload(file=jsonl_path, config={"mime_type": "application/jsonl"})

    click.echo("Submitting batch job...")
    batch_job = client.batches.create(
        model=f"models/{batch_config['gemini_model']}",
        src=uploaded_file.name,
        config=types.CreateBatchJobConfig(display_name=f"nano_banana_batch_{int(time.time())}"),
    )

    # Save batch info
    batch_info_path = output_dir / "batch_info.json"
    with open(batch_info_path, "w") as f:
        json.dump(
            {
                "batch_name": batch_job.name,
                "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )

    click.echo(f"Batch job submitted: {batch_job.name}")
    click.echo(f"Saved batch info to {batch_info_path}")
    click.echo(f"\nCheck status: python run_nano_banana_batch.py retrieve --output-dir {output_dir}")


@cli.command()
@click.option(
    "--output-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Output directory with batch_info.json",
)
def retrieve(output_dir: Path) -> None:
    """Check batch status and retrieve results if completed."""
    batch_info_path = output_dir / "batch_info.json"

    if not batch_info_path.exists():
        click.echo(f"Error: {batch_info_path} not found. Run 'submit' first.")
        return

    with open(batch_info_path) as f:
        batch_info = json.load(f)

    config = PipelineConfig()
    client = genai.Client(api_key=config.google_api_key)

    batch_name = batch_info["batch_name"]
    click.echo(f"Checking batch job: {batch_name}")

    batch_job = client.batches.get(name=batch_name)
    state = batch_job.state.name if batch_job.state else "UNKNOWN"
    click.echo(f"Status: {state}")

    if state not in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
        click.echo("Batch job still running. Check again later.")
        return

    if state != "JOB_STATE_SUCCEEDED":
        click.echo(f"Batch job did not succeed: {state}")
        return

    # Retrieve results
    click.echo("Retrieving results...")
    images_dir = output_dir / "generated_images"
    images_dir.mkdir(exist_ok=True)

    results: list[dict] = []

    # Get results based on destination type
    if hasattr(batch_job.dest, "inlined_responses") and batch_job.dest.inlined_responses:
        responses = batch_job.dest.inlined_responses
    elif hasattr(batch_job.dest, "file_name") and batch_job.dest.file_name:
        result_file = client.files.download(file=batch_job.dest.file_name)
        responses = [json.loads(line) for line in result_file.decode("utf-8").strip().split("\n")]
    else:
        click.echo("No results found in batch job")
        return

    import base64

    for response in responses:
        key = response.get("key", "")
        idx = int(key.replace("request-", "")) if key.startswith("request-") else -1

        if "error" in response:
            results.append({"index": idx, "image_path": None, "error": response["error"]})
            continue

        try:
            content = response.get("response", {}).get("candidates", [{}])[0].get("content", {})
            parts = content.get("parts", [])

            image_bytes = None
            for part in parts:
                if "inlineData" in part:
                    image_bytes = base64.b64decode(part["inlineData"]["data"])
                    break

            if image_bytes:
                image_path = images_dir / f"generated_{idx}.png"
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                results.append({"index": idx, "image_path": str(image_path), "error": None})
            else:
                results.append({"index": idx, "image_path": None, "error": "No image in response"})

        except Exception as e:
            results.append({"index": idx, "image_path": None, "error": str(e)})

    # Save results
    results_path = output_dir / "image_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    saved = sum(1 for r in results if r["image_path"] is not None)
    failed = sum(1 for r in results if r["error"] is not None)
    click.echo(f"Done: {saved} images saved, {failed} failed")
    click.echo(f"Results saved to {results_path}")
    click.echo(f"Images saved to {images_dir}")


def save_full_results(
    outputs: list[tuple[int, PipelineResult | None, str | None]],
    output_dir: Path,
) -> tuple[int, int]:
    """Save full pipeline results including generated images.

    Returns:
        Tuple of (success_count, failure_count).
    """
    images_dir = output_dir / "generated_images"
    images_dir.mkdir(exist_ok=True)

    results: list[dict] = []
    success_count = 0
    failure_count = 0

    for idx, output, error in outputs:
        if output is None or error is not None:
            results.append({
                "index": idx,
                "output": None,
                "image_path": None,
                "error": error,
            })
            failure_count += 1
            continue

        # Save generated image if present (full pipeline output)
        image_path = None
        if output.get("generated_image_bytes"):
            image_path = images_dir / f"generated_{idx}.png"
            with open(image_path, "wb") as f:
                f.write(output["generated_image_bytes"])
            image_path = str(image_path)

        # Create serializable output (exclude raw image bytes)
        serializable_output = {k: v for k, v in dict(output).items() if k != "generated_image_bytes"}

        results.append({
            "index": idx,
            "output": serializable_output,
            "image_path": image_path,
            "error": None,
        })
        success_count += 1

    # Save results JSON
    results_path = output_dir / "full_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return success_count, failure_count


@cli.command()
@click.option(
    "--input-excel",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input Excel file with dataset",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("output/nano_banana_full"),
    help="Output directory for results",
)
@click.option("--image-path-col", type=str, default="image_path")
@click.option("--visual-desc-col", type=str, default="visual_description")
@click.option("--max-concurrent", type=int, default=3, help="Max concurrent requests (lower for image generation)")
def run_all(
    input_excel: Path,
    output_dir: Path,
    image_path_col: str,
    visual_desc_col: str,
    max_concurrent: int,
) -> None:
    """Run the full pipeline (Steps 1-5) for all items.

    This runs Steps 1-5 synchronously without batch API.
    Suitable for smaller datasets or when immediate results are needed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    click.echo(f"Loading dataset from {input_excel}...")
    df = pd.read_excel(input_excel)
    click.echo(f"Loaded {len(df)} rows")

    # Initialize full pipeline (with image generation)
    config = PipelineConfig(skip_image_generation=False)
    pipeline = NanoBananaPipeline(config)

    # Process all items through Steps 1-5
    click.echo("Running Steps 1-5 (full pipeline with image generation)...")
    start_time = time.time()

    outputs = asyncio.run(
        process_dataset(
            df=df,
            pipeline=pipeline,
            image_path_col=image_path_col,
            visual_desc_col=visual_desc_col,
            max_concurrent=max_concurrent,
        )
    )

    elapsed = time.time() - start_time
    click.echo(f"Pipeline complete in {elapsed:.1f}s")

    # Save results
    click.echo("Saving results...")
    success_count, failure_count = save_full_results(outputs, output_dir)

    click.echo(f"\nDone: {success_count} successes, {failure_count} failures")
    click.echo(f"Results saved to {output_dir / 'full_results.json'}")
    if success_count > 0:
        click.echo(f"Images saved to {output_dir / 'generated_images'}")


if __name__ == "__main__":
    load_dotenv()
    cli()
