#!/usr/bin/env python
"""
batch_runner.py

Split WAV files into batches and invoke the inference CLI for each batch,
showing progress over batches.
"""

import os
import math
import subprocess
import logging
from pathlib import Path

from tqdm.auto import tqdm
from config import MODEL_DIR, INPUT_DIR, OUTPUT_DIR, BATCH_SIZE, SCRATCH_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def main() -> None:
    """
    Main entrypoint: batch up your .wav files and call the inference CLI for each batch.
    Shows a progress bar over batches.
    """
    # ensure output and scratch directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    # gather all WAV filenames
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".wav")]
    total_files = len(files)
    if total_files == 0:
        logging.error("No WAV files found in %s", INPUT_DIR)
        return

    total_batches = math.ceil(total_files / BATCH_SIZE)
    logging.info(
        "Found %d files; will process in %d batches (batch size=%d)",
        total_files, total_batches, BATCH_SIZE
    )

    # batch-level progress bar
    for batch_idx in tqdm(range(total_batches), desc="Batches", unit="batch"):
        start = batch_idx * BATCH_SIZE
        batch_files = files[start:start + BATCH_SIZE]

        # write this batch list into scratch
        batch_file = SCRATCH_DIR / f"batch_{batch_idx}.txt"
        batch_file.write_text("\n".join(batch_files))

        logging.info(
            "Starting batch %d/%d with %d files",
            batch_idx + 1, total_batches, len(batch_files)
        )

        # invoke the per-batch inference CLI
        subprocess.run(
            [
                "python",
                "-m", "scripts.run_inference",
                "--model-dir",  str(MODEL_DIR),
                "--input-dir",  str(INPUT_DIR),
                "--output-dir", str(OUTPUT_DIR),
                "--batch-file", str(batch_file)
            ],
            check=True
        )

        # clean up
        batch_file.unlink()
        logging.info("Finished batch %d/%d", batch_idx + 1, total_batches)

    logging.info("All %d batches completed", total_batches)


if __name__ == "__main__":
    main()
