"""Dataset classes for voice conversion training."""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from .audio import (
    load_audio, extract_mel_spectrogram, extract_f0,
    segment_audio, normalize_audio, augment_audio
)


class VoiceConversionDataset(Dataset):
    """Dataset for voice conversion training.
    
    Expects directory structure:
        data_dir/
            speaker1/
                audio1.wav
                audio2.wav
                ...
            speaker2/
                ...
    """
    
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 44100,
        segment_length: int = 8192,
        hop_length: int = 512,
        n_fft: int = 2048,
        n_mels: int = 80,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        augmentation: Optional[dict] = None,
        cache: bool = False
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Root directory with speaker subdirectories
            sample_rate: Target sample rate
            segment_length: Length of audio segments in samples
            hop_length: Hop length for spectrogram
            n_fft: FFT window size
            n_mels: Number of mel bins
            f0_min: Minimum F0 frequency
            f0_max: Maximum F0 frequency
            augmentation: Augmentation config dict
            cache: Cache loaded audio in memory
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.augmentation = augmentation or {}
        self.cache = cache
        
        # Find all audio files
        self.audio_files = []
        self.speaker_ids = {}
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        # Scan for speakers and audio files
        speaker_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if not speaker_dirs:
            # If no subdirectories, treat all files as one speaker
            audio_files = list(self.data_dir.glob("*.wav")) + list(self.data_dir.glob("*.mp3"))
            if audio_files:
                self.audio_files = [(str(f), 0) for f in audio_files]
                self.speaker_ids = {0: "default"}
        else:
            # Multiple speakers
            for speaker_idx, speaker_dir in enumerate(sorted(speaker_dirs)):
                self.speaker_ids[speaker_idx] = speaker_dir.name
                audio_files = (
                    list(speaker_dir.glob("*.wav")) +
                    list(speaker_dir.glob("*.mp3"))
                )
                for audio_file in audio_files:
                    self.audio_files.append((str(audio_file), speaker_idx))
        
        if not self.audio_files:
            raise ValueError(f"No audio files found in {data_dir}")
        
        print(f"Found {len(self.audio_files)} audio files from {len(self.speaker_ids)} speakers")
        
        # Cache for loaded audio
        self._audio_cache = {}
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def _load_audio(self, path: str) -> np.ndarray:
        """Load audio, with optional caching."""
        if self.cache and path in self._audio_cache:
            return self._audio_cache[path]
        
        audio, sr = load_audio(path, self.sample_rate, mono=True)
        audio = normalize_audio(audio)
        
        if self.cache:
            self._audio_cache[path] = audio
        
        return audio
    
    def __getitem__(self, idx: int) -> dict:
        """Get a training sample.
        
        Returns:
            Dictionary with:
                - audio: [segment_length] audio segment
                - mel: [n_mels, time] mel spectrogram
                - f0: [time] F0 values
                - speaker_id: speaker ID
        """
        audio_path, speaker_id = self.audio_files[idx]
        
        # Load audio
        audio = self._load_audio(audio_path)
        
        # Apply augmentation if enabled
        if self.augmentation.get("enabled", False):
            pitch_shift = None
            time_stretch = None
            noise_level = 0.0
            
            if "pitch_shift_range" in self.augmentation:
                pitch_range = self.augmentation["pitch_shift_range"]
                pitch_shift = random.uniform(pitch_range[0], pitch_range[1])
            
            if "time_stretch_range" in self.augmentation:
                stretch_range = self.augmentation["time_stretch_range"]
                time_stretch = random.uniform(stretch_range[0], stretch_range[1])
            
            if "noise_level" in self.augmentation:
                noise_level = self.augmentation["noise_level"]
            
            audio = augment_audio(
                audio,
                self.sample_rate,
                pitch_shift=pitch_shift,
                time_stretch=time_stretch,
                noise_level=noise_level
            )
        
        # Extract random segment
        if len(audio) < self.segment_length:
            # Pad if too short
            audio = np.pad(audio, (0, self.segment_length - len(audio)), mode="constant")
        else:
            # Random crop
            start = random.randint(0, len(audio) - self.segment_length)
            audio = audio[start:start + self.segment_length]
        
        # Extract features
        mel = extract_mel_spectrogram(
            audio,
            self.sample_rate,
            self.n_fft,
            self.hop_length,
            self.n_mels
        )
        
        f0 = extract_f0(
            audio,
            self.sample_rate,
            self.f0_min,
            self.f0_max,
            self.hop_length
        )
        
        # Convert to tensors
        audio_tensor = torch.FloatTensor(audio)
        mel_tensor = torch.FloatTensor(mel)
        f0_tensor = torch.FloatTensor(f0)
        speaker_tensor = torch.LongTensor([speaker_id])
        
        return {
            "audio": audio_tensor,
            "mel": mel_tensor,
            "f0": f0_tensor,
            "speaker_id": speaker_tensor.squeeze()
        }

