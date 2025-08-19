#!/usr/bin/env python3
"""
Feature extraction for bomb detection model
Extracts MFCC features from augmented training data and test data
Based on the notebook code provided by the user
"""

import datetime
import numpy as np
import random
import os
import pickle
from pathlib import Path

# Progress tracker for feature extraction
from tqdm import tqdm

# For audio processing
import librosa
import librosa.display

# For visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# Set up matplotlib
mpl.rcParams["figure.figsize"] = (12, 10)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class FeatureExtractor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.final_dataset_dir = self.data_dir / "final_new_dataset"

        # Directory paths
        self.train_augmented_dir = self.final_dataset_dir / "train_augmented"
        self.test_dir = self.final_dataset_dir / "test"

        # Sample rate (matching the processed audio files)
        self.sample_rate = 8000

        # Set random seed for reproducibility
        self.seed = 123
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Output pickle files
        self.train_pickle_file = "/Users/sonnyburniston/Bomb-Fishing/data/new_data_train_features_labels.pickle"
        self.test_pickle_file = "/Users/sonnyburniston/Bomb-Fishing/data/new_data_test_features_labels.pickle"

        # Verify directories exist
        if not self.train_augmented_dir.exists():
            raise FileNotFoundError(
                f"Augmented training directory not found: {self.train_augmented_dir}"
            )
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")

    def view_mel_spec(self, filename: str, audio_dir: Path):
        """Creates a spectrogram plot of an audio file in the mel scale.

        Args:
            filename: Audio filename
            audio_dir: Directory containing the audio file
        """
        # Set hop length for mel_spec and specshow()
        hop_length = 64

        # Read in the audio file
        file_path = audio_dir / filename
        audio, sample_rate = librosa.load(file_path, sr=self.sample_rate)

        # Compute the mel spectrogram of the audio
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_mels=128, win_length=1024, hop_length=hop_length
        )

        # Convert the power spectrogram to decibel (dB) units
        mel_spec_db = librosa.power_to_db(S=mel_spec, ref=1.0)

        # Plot the spectrogram
        plt.figure(figsize=(5, 5))
        img = librosa.display.specshow(
            mel_spec_db,
            sr=sample_rate,
            y_axis="mel",
            x_axis="time",
            vmin=-100,
            vmax=0,
            cmap="magma",
            hop_length=hop_length,
        )
        plt.colorbar(img)

        # Set y-axis
        plt.yticks([200, 500, 1000, 2000, 4000])
        plt.ylim(0, 4000)
        plt.title(f"Mel Spectrogram: {filename}")

    def extract_features_labels(self, dataset: list, audio_dir: Path):
        """Extract MFCC features and labels from audio files.

        Args:
            dataset: List of audio filenames
            audio_dir: Directory containing the audio files

        Returns:
            features: Numpy array of MFCC features
            labels: Numpy array of labels (0 for non-bomb, 1 for bomb)
            input_shape: Shape of input features for the network
        """
        # Create empty lists
        feature_list = []
        label_list = []

        # Extract MFCC from all files
        for file in tqdm(dataset, desc="Extracting features"):
            # Load audio
            audio_path = audio_dir / file
            audio, sr = librosa.load(path=audio_path, sr=self.sample_rate)

            # Calculate MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=32)

            # Define new shape
            shape = np.shape(mfcc)
            input_shape = tuple(list(shape) + [1])

            # Set new shape and append to list
            feature_list.append(np.reshape(mfcc, input_shape))

            # Get label from first character of file name and append to list
            if file[0:2] == "NB":
                label_list.append([0])  # Non-bomb
            elif file[0:2] == "YB":
                label_list.append([1])  # Bomb
            else:
                raise ValueError(f"Unexpected file prefix in {file}")

        # Convert to numpy arrays which can be input into the network
        features = np.array(feature_list)
        labels = np.array(label_list)

        return features, labels, input_shape

    def custom_sort_key(self, string: str):
        """Custom sorting key for test files - NB files first, then YB files."""
        if string.startswith("NB"):
            return (0, int(string[2:7]))
        elif string.startswith("YB"):
            return (1, int(string[2:7]))
        else:
            raise ValueError(f"Unexpected string format: {string}")

    def plot_mfcc(self, features: np.ndarray, chosen_file: int, title: str = "MFCC"):
        """Plot MFCC features for a specific file."""
        # Drop the extra channel added for the network
        spec_data = features[chosen_file][:, :, 0]

        # Create a figure with specified width and height
        fig = plt.figure(figsize=(8, 5))

        # Add subplot to the figure
        ax = fig.add_subplot(111)

        # Plot the data
        ax.imshow(
            spec_data,
            interpolation="nearest",
            cmap="magma",
            origin="lower",
            vmin=-100,
            vmax=0.0,
        )
        ax.set_title(f"{title}: File {chosen_file}")

        # Set aspect ratio to the aspect ratio of the data
        aspect = spec_data.shape[1] / spec_data.shape[0]
        ax.set_aspect(aspect)

        plt.show()

    def extract_train_features(self):
        """Extract features from augmented training data."""
        print("=== EXTRACTING TRAINING FEATURES ===")

        # Make list of all audio files in the directory
        train_files = [
            f
            for f in os.listdir(self.train_augmented_dir)
            if f.endswith(".wav") or f.endswith(".WAV")
        ]
        train_file_count = len(train_files)
        print(f"Found {train_file_count} training files")

        # Shuffle list of files so files from same sites get shuffled around
        random.shuffle(train_files)

        # Extract features and labels, plus save input shape for network
        train_features, train_labels, input_shape = self.extract_features_labels(
            train_files, self.train_augmented_dir
        )

        # Print statistics
        print(f"\nInput shape for network: {input_shape}")
        bomb_indices = np.where(train_labels == 1)[0]
        non_bomb_indices = np.where(train_labels == 0)[0]
        print(f"Bomb files: {len(bomb_indices)}")
        print(f"Non-bomb files: {len(non_bomb_indices)}")
        print(f"Bomb file indices: {bomb_indices[:10]}...")  # Show first 10

        # Save to pickle file
        pickle_file_path = Path(self.train_pickle_file)
        with open(pickle_file_path, "wb") as f:
            pickle.dump((train_features, train_labels, input_shape), f)

        print(f"Training features saved to: {pickle_file_path}")

        return train_features, train_labels, input_shape

    def extract_test_features(self):
        """Extract features from test data."""
        print("\n=== EXTRACTING TEST FEATURES ===")

        # List all audio files in the directory
        test_files = [
            f
            for f in os.listdir(self.test_dir)
            if f.endswith(".wav") or f.endswith(".WAV")
        ]
        test_file_count = len(test_files)
        print(f"Found {test_file_count} test files")

        # Sort test files (NB files first, then YB files)
        sorted_test_files = sorted(test_files, key=self.custom_sort_key)

        # Extract features and labels
        test_features, test_labels, input_shape = self.extract_features_labels(
            sorted_test_files, self.test_dir
        )

        # Print statistics
        print(f"\nInput shape for network: {input_shape}")
        bomb_indices = np.where(test_labels == 1)[0]
        non_bomb_indices = np.where(test_labels == 0)[0]
        print(f"Bomb files: {len(bomb_indices)}")
        print(f"Non-bomb files: {len(non_bomb_indices)}")
        print(f"Bomb file indices: {bomb_indices}")

        # Save to pickle file
        pickle_file_path = Path(self.test_pickle_file)
        with open(pickle_file_path, "wb") as f:
            pickle.dump((test_features, test_labels, input_shape), f)

        print(f"Test features saved to: {pickle_file_path}")

        return test_features, test_labels, input_shape

    def load_pickle_file(self, pickle_file: str):
        """Load features and labels from pickle file."""
        pickle_file_path = Path(pickle_file)

        if not pickle_file_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {pickle_file_path}")

        with open(pickle_file_path, "rb") as f:
            features, labels, input_shape = pickle.load(f)

        return features, labels, input_shape

    def create_metadata(self, train_features, train_labels, test_features, test_labels):
        """Create metadata file documenting the feature extraction."""
        metadata_file = Path("feature_extraction_metadata.txt")

        with open(metadata_file, "w") as f:
            f.write("FEATURE EXTRACTION METADATA\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write("FEATURE EXTRACTION PARAMETERS:\n")
            f.write(f"  Sample rate: {self.sample_rate} Hz\n")
            f.write("  MFCC features: 32\n")
            f.write(f"  Random seed: {self.seed}\n\n")

            f.write("TRAINING DATA (AUGMENTED):\n")
            f.write(f"  Total files: {len(train_features)}\n")
            f.write(f"  Bomb files: {np.sum(train_labels)}\n")
            f.write(f"  Non-bomb files: {len(train_labels) - np.sum(train_labels)}\n")
            f.write(f"  Feature shape: {train_features.shape}\n\n")

            f.write("TEST DATA:\n")
            f.write(f"  Total files: {len(test_features)}\n")
            f.write(f"  Bomb files: {np.sum(test_labels)}\n")
            f.write(f"  Non-bomb files: {len(test_labels) - np.sum(test_labels)}\n")
            f.write(f"  Feature shape: {test_features.shape}\n\n")

            f.write("OUTPUT FILES:\n")
            f.write(f"  Training features: {self.train_pickle_file}\n")
            f.write(f"  Test features: {self.test_pickle_file}\n\n")

            f.write("NOTES:\n")
            f.write(
                "  - Features are MFCC spectrograms with shape (32, time_steps, 1)\n"
            )
            f.write("  - Labels: 0 = non-bomb (NB), 1 = bomb (YB)\n")
            f.write("  - Training data includes augmented samples\n")

        print(f"Metadata saved to: {metadata_file}")

    def run_feature_extraction(self):
        """Run the complete feature extraction pipeline."""
        print("Starting feature extraction pipeline...")

        # Extract training features
        train_features, train_labels, input_shape = self.extract_train_features()

        # Extract test features
        test_features, test_labels, input_shape = self.extract_test_features()

        # Create metadata
        self.create_metadata(train_features, train_labels, test_features, test_labels)

        print("\n=== FEATURE EXTRACTION COMPLETE ===")
        print(f"Training features: {train_features.shape}")
        print(f"Test features: {test_features.shape}")
        print(f"Input shape for network: {input_shape}")

        return train_features, train_labels, test_features, test_labels, input_shape


def main():
    """Run the feature extraction pipeline."""
    extractor = FeatureExtractor()
    extractor.run_feature_extraction()


if __name__ == "__main__":
    main()
