# inference/bomb_detector.py

import os
import datetime
import logging
import numpy as np
import soundfile as sf
from typing import List, Tuple, Any, Optional

from inference.audio_utils import (
    find_audio_files, load_and_resample, sliding_windows, compute_mfcc
)
from inference.model_utils import load_bomb_model

logger = logging.getLogger(__name__)


class BombDetector:
  """Encapsulates bomb inference over audio files with batched predictions."""

  def __init__(
    self,
    model_dir: str,
    input_dir: str,
    output_dir: str
  ) -> None:
    """
    Args:
        model_dir: Path to saved TensorFlow model.
        input_dir: Directory containing .wav files.
        output_dir: Directory to save detected clips.
    """
    self.model = load_bomb_model(model_dir)
    self.input_dir = input_dir
    self.output_dir = output_dir
    os.makedirs(self.output_dir, exist_ok=True)

  def run_inference(
    self,
    files: Optional[List[str]] = None,
    batch_size: int = 32
  ) -> List[Tuple[str, str]]:
    """
    Process files (or all if None), batch MFCC windows for each file,
    run model predict in batches, and save detected bomb clips.

    Args:
        files: List of wav filenames to process.
        batch_size: Batch size for model.predict.

    Returns:
        List of tuples (filename, timestamp) for suspected bombs.
    """
    files_to_check = files or find_audio_files(self.input_dir)
    results: List[Tuple[str, str]] = []

    for fname in files_to_check:
      filepath = os.path.join(self.input_dir, fname)
      audio, sr = load_and_resample(filepath)
      windows = sliding_windows(audio, sr)

      # Compute all MFCCs and stack into a single batch to pass to GPU
      mfcc_list = [compute_mfcc(win, sr)[0] for win, _ in windows]
      mfcc_batch = np.stack(mfcc_list, axis=0)

      # Inference wiht model on batch
      probs = self.model.predict(
        mfcc_batch, batch_size=batch_size, verbose=0
      ).flatten()

      # Process results
      for (win, start_time), prob in zip(windows, probs):
        if prob > 0.5:
          ts = datetime.timedelta(seconds=start_time)
          ts_str = str(ts)[:7].replace(":", ".")
          clip = self._extract_clip(audio, sr, start_time)
          out_name = f"{fname[:-4]}_{ts_str}.wav"
          out_path = os.path.join(self.output_dir, out_name)
          sf.write(out_path, clip, sr)
          results.append((fname, ts_str))
          logger.info("Detected bomb in %s at %s", fname, ts)

    return results

  def _extract_clip(
    self,
    audio: Any,
    sr: int,
    start_time: float
  ) -> Any:
    """Return 5s snippet around start_time (1s before, 4s after)."""
    start_idx = max(int((start_time - 1) * sr), 0)
    end_idx = min(int((start_time + 4) * sr), len(audio))
    return audio[start_idx:end_idx]
