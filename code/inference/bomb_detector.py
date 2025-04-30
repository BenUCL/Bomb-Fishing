# inference/bomb_detector.py

import os
import datetime
import logging
from typing import List, Tuple, Any, Optional

import numpy as np
import soundfile as sf

from config import STAGGER_SEC
from inference.audio_utils import (
    find_audio_files, load_and_resample, sliding_windows, compute_mfcc
)
from inference.model_utils import load_bomb_model

logger = logging.getLogger(__name__)


class BombDetector:
    """Encapsulates batched bomb inference with duplicate-detection suppression."""

    def __init__(
        self,
        model_dir: str,
        input_dir: str,
        output_dir: str
    ) -> None:
        """
        Args:
            model_dir: path to saved TF model
            input_dir: directory of .wav files
            output_dir: where to write detected clips
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
        For each file, batch-compute MFCCs, do one model.predict,
        then de-dupe overlapping hits and save one clip per event.

        Args:
            files: specific filenames to process (else all in input_dir)
            batch_size: batch size for model.predict

        Returns:
            List of (filename, timestamp_str) for each unique bomb.
        """
        files_to_check = files or find_audio_files(self.input_dir)
        results: List[Tuple[str, str]] = []

        for fname in files_to_check:
            filepath = os.path.join(self.input_dir, fname)
            audio, sr = load_and_resample(filepath)
            windows = sliding_windows(audio, sr)

            # 1) batch compute MFCCs
            mfcc_list = [compute_mfcc(win, sr)[0] for win, _ in windows]
            mfcc_batch = np.stack(mfcc_list, axis=0)

            # 2) single bulk predict
            probs = self.model.predict(
                mfcc_batch, batch_size=batch_size, verbose=0
            ).flatten()

            # 3) collect raw detection times
            raw_times = [
                start_time
                for (_, start_time), p in zip(windows, probs)
                if p > 0.5
            ]
            raw_times.sort()

            # 4) de-duplicate times closer than STAGGER_SEC
            unique_times: List[float] = []
            for t in raw_times:
                if not unique_times or (t - unique_times[-1]) > STAGGER_SEC:
                    unique_times.append(t)

            # 5) save one clip per unique detection
            for start_time in unique_times:
                ts = datetime.timedelta(seconds=start_time)
                hours   = int(start_time // 3600)
                minutes = int((start_time % 3600) // 60)
                seconds = int(start_time % 60)
                ts_str  = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
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
        """
        Return a 5s snippet around the bomb: 1s before to 4s after.
        """
        start_idx = max(int((start_time - 1) * sr), 0)
        end_idx   = min(int((start_time + 4) * sr), len(audio))
        return audio[start_idx:end_idx]
