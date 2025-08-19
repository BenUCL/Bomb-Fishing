#!/usr/bin/env python3
"""
Create train/test split for the processed data
Based on hold-out months: 2023_aug_03, 2023_nov_23, 2024_apr_24
Currently using 2023_nov_23 as test set (others to be uploaded later)
"""

import os
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrainTestSplitter:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed_new_data"
        self.final_dir = self.data_dir / "final_new_dataset"

        # Define train/test split
        self.test_months = [
            "2023_aug_21",
            "2023_nov_23",
            "2024_march_12",
        ]
        self.current_test_months = [
            "2023_aug_21",
            "2023_nov_23",
            "2024_mar_12",
        ]  # Only nov_23 available now

        # Create output directories
        self.train_dir = self.final_dir / "train"
        self.test_dir = self.final_dir / "test"

        for dir_path in [self.train_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_available_months(self):
        """Get list of available months in processed data."""
        if not self.processed_dir.exists():
            logger.error(f"Processed data directory not found: {self.processed_dir}")
            return []

        months = [d.name for d in self.processed_dir.iterdir() if d.is_dir()]
        logger.info(f"Available months: {months}")
        return months

    def count_files_by_type(self, month_dir: Path):
        """Count YB and NB files in a month directory."""
        yb_count = len(list(month_dir.glob("YB*.wav")))
        nb_count = len(list(month_dir.glob("NB*.wav")))
        return yb_count, nb_count

    def analyze_data_distribution(self):
        """Analyze the distribution of data across months."""
        logger.info("=== DATA DISTRIBUTION ANALYSIS ===")

        available_months = self.get_available_months()
        total_yb = 0
        total_nb = 0

        for month in available_months:
            month_dir = self.processed_dir / month
            yb_count, nb_count = self.count_files_by_type(month_dir)
            total_yb += yb_count
            total_nb += nb_count

            logger.info(
                f"{month}: {yb_count} bombs, {nb_count} non-bombs ({yb_count + nb_count} total)"
            )

        logger.info(
            f"TOTAL: {total_yb} bombs, {total_nb} non-bombs ({total_yb + total_nb} total)"
        )

        # Identify train vs test months
        train_months = [
            m for m in available_months if m not in self.current_test_months
        ]
        test_months = [m for m in available_months if m in self.current_test_months]

        logger.info(f"\nTrain months: {train_months}")
        logger.info(f"Test months: {test_months}")

        return train_months, test_months

    def copy_files_to_split(self, source_dir: Path, target_dir: Path, file_type: str):
        """Copy files of a specific type to the target directory."""
        files = list(source_dir.glob(f"{file_type}*.wav"))
        for file in files:
            target_file = target_dir / file.name
            shutil.copy2(file, target_file)
        return len(files)

    def create_split(self):
        """Create the train/test split."""
        logger.info("=== CREATING TRAIN/TEST SPLIT ===")

        # Analyze current data distribution
        train_months, test_months = self.analyze_data_distribution()

        if not train_months:
            logger.error("No training months available!")
            return

        if not test_months:
            logger.warning("No test months available! Using last month as test.")
            test_months = [train_months[-1]]
            train_months = train_months[:-1]

        # Clear existing directories
        for split_dir in [self.train_dir, self.test_dir]:
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)

        # Copy training data
        logger.info(f"\nCopying training data from {len(train_months)} months...")
        train_yb_total = 0
        train_nb_total = 0

        for month in train_months:
            month_dir = self.processed_dir / month
            if month_dir.exists():
                yb_count = self.copy_files_to_split(month_dir, self.train_dir, "YB")
                nb_count = self.copy_files_to_split(month_dir, self.train_dir, "NB")
                train_yb_total += yb_count
                train_nb_total += nb_count
                logger.info(f"  {month}: {yb_count} bombs, {nb_count} non-bombs")

        # Copy test data
        logger.info(f"\nCopying test data from {len(test_months)} months...")
        test_yb_total = 0
        test_nb_total = 0

        for month in test_months:
            month_dir = self.processed_dir / month
            if month_dir.exists():
                yb_count = self.copy_files_to_split(month_dir, self.test_dir, "YB")
                nb_count = self.copy_files_to_split(month_dir, self.test_dir, "NB")
                test_yb_total += yb_count
                test_nb_total += nb_count
                logger.info(f"  {month}: {yb_count} bombs, {nb_count} non-bombs")

        # Summary
        logger.info(f"\n=== SPLIT SUMMARY ===")
        logger.info(
            f"Train: {train_yb_total} bombs, {train_nb_total} non-bombs ({train_yb_total + train_nb_total} total)"
        )
        logger.info(
            f"Test:  {test_yb_total} bombs, {test_nb_total} non-bombs ({test_yb_total + test_nb_total} total)"
        )
        logger.info(
            f"Total: {train_yb_total + test_yb_total} bombs, {train_nb_total + test_nb_total} non-bombs"
        )

        # Calculate ratios
        total_files = train_yb_total + train_nb_total + test_yb_total + test_nb_total
        train_ratio = (train_yb_total + train_nb_total) / total_files
        test_ratio = (test_yb_total + test_nb_total) / total_files

        logger.info(f"Split ratio: {train_ratio:.1%} train, {test_ratio:.1%} test")

        # Verify file counts
        train_files = len(list(self.train_dir.glob("*.wav")))
        test_files = len(list(self.test_dir.glob("*.wav")))

        logger.info(f"Verification: {train_files} train files, {test_files} test files")

        return {
            "train_months": train_months,
            "test_months": test_months,
            "train_yb": train_yb_total,
            "train_nb": train_nb_total,
            "test_yb": test_yb_total,
            "test_nb": test_nb_total,
        }

    def create_metadata_file(self, split_info: dict):
        """Create a metadata file documenting the split."""
        metadata_file = self.final_dir / "split_metadata.txt"

        with open(metadata_file, "w") as f:
            f.write("TRAIN/TEST SPLIT METADATA\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Created: {split_info.get('timestamp', 'N/A')}\n\n")
            f.write("SPLIT CONFIGURATION:\n")
            f.write(f"  Test months (hold-out): {split_info['test_months']}\n")
            f.write(f"  Train months: {split_info['train_months']}\n\n")
            f.write("DATA COUNTS:\n")
            f.write(f"  Train bombs (YB): {split_info['train_yb']}\n")
            f.write(f"  Train non-bombs (NB): {split_info['train_nb']}\n")
            f.write(
                f"  Train total: {split_info['train_yb'] + split_info['train_nb']}\n\n"
            )
            f.write(f"  Test bombs (YB): {split_info['test_yb']}\n")
            f.write(f"  Test non-bombs (NB): {split_info['test_nb']}\n")
            f.write(
                f"  Test total: {split_info['test_yb'] + split_info['test_nb']}\n\n"
            )
            f.write("NOTES:\n")
            f.write(
                "  - Hold-out months 2023_aug_03 and 2024_apr_24 will be added later\n"
            )
            f.write("  - Currently using 2023_nov_23 as test set\n")
            f.write("  - All files are 2.88s duration at 8kHz sample rate\n")

        logger.info(f"Metadata saved to: {metadata_file}")


def main():
    """Run the train/test split creation."""
    splitter = TrainTestSplitter()
    split_info = splitter.create_split()

    if split_info:
        splitter.create_metadata_file(split_info)
        logger.info("Train/test split creation complete!")
    else:
        logger.error("Failed to create train/test split!")


if __name__ == "__main__":
    main()
