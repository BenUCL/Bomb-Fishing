#!/usr/bin/env python
"""
run_inference.py

CLI to run bomb detection on a batch of WAV files, showing a progress bar per batch.
"""

import argparse
import logging
from pathlib import Path

from tqdm.auto import tqdm
import pandas as pd

from config import MODEL_DIR, INPUT_DIR, OUTPUT_DIR
from inference.bomb_detector import BombDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def main() -> None:
    """
    Load the list of files to process from a batch file, run inference on each
    with a progress bar, and save the results as a CSV.
    """
    parser = argparse.ArgumentParser(
        description="Run bomb detection on a batch of WAV files."
    )
    parser.add_argument("--model-dir",  default=str(MODEL_DIR))
    parser.add_argument("--input-dir",  default=str(INPUT_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--batch-file",
        required=True,
        help="Text file listing WAV filenames (one basename per line)."
    )
    args = parser.parse_args()

    batch_list = Path(args.batch_file).read_text().splitlines()
    if not batch_list:
        logging.warning("Batch file %s is empty.", args.batch_file)
        return

    detector = BombDetector(
        model_dir=args.model_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )

    all_results = []
    # file-level progress bar
    for fname in tqdm(batch_list, desc="Files", unit="file"):
        # run inference on just this one file
        results = detector.run_inference(files=[fname])
        all_results.extend(results)

    # write out a per-batch CSV
    df = pd.DataFrame(all_results, columns=["File", "Timestamp"])
    csv_name = Path(args.batch_file).stem + "_results.csv"
    csv_path = Path(args.output_dir) / csv_name
    df.to_csv(csv_path, index=False)
    logging.info("Saved %d detections to %s", len(df), csv_path)


if __name__ == "__main__":
    main()
