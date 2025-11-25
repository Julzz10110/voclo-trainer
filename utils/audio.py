"""Audio processing utilities for voice conversion training."""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, List
import torch
import torchaudio


def load_audio(
    path: str,
    sample_rate: int = 44100,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """Load audio file and resample if needed.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio_array, actual_sample_rate)
    """
    audio, sr = librosa.load(path, sr=sample_rate, mono=mono)
    return audio, sr


def save_audio(
    path: str,
    audio: np.ndarray,
    sample_rate: int = 44100
) -> None:
    """Save audio to file.
    
    Args:
        path: Output path
        audio: Audio array (mono or stereo)
        sample_rate: Sample rate
    """
    sf.write(path, audio, sample_rate)


def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    """Extract mel spectrogram from audio.
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bins
        fmin: Minimum frequency
        fmax: Maximum frequency (None = sample_rate / 2)
        
    Returns:
        Mel spectrogram [n_mels, time]
    """
    if fmax is None:
        fmax = sample_rate / 2
    
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    
    # Convert to log scale
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    return mel_db


def extract_f0(
    audio: np.ndarray,
    sample_rate: int = 44100,
    fmin: float = 50.0,
    fmax: float = 1100.0,
    hop_length: int = 512
) -> np.ndarray:
    """Extract fundamental frequency (F0) using PYIN algorithm.
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate
        fmin: Minimum frequency
        fmax: Maximum frequency
        hop_length: Hop length
        
    Returns:
        F0 values [time]
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        hop_length=hop_length
    )
    
    # Replace NaN with 0
    f0 = np.nan_to_num(f0, nan=0.0)
    
    return f0


def segment_audio(
    audio: np.ndarray,
    segment_length: int,
    hop_length: Optional[int] = None
) -> List[np.ndarray]:
    """Segment audio into fixed-length chunks.
    
    Args:
        audio: Audio signal
        segment_length: Length of each segment in samples
        hop_length: Hop between segments (default: segment_length)
        
    Returns:
        List of audio segments
    """
    if hop_length is None:
        hop_length = segment_length
    
    segments = []
    for start in range(0, len(audio) - segment_length + 1, hop_length):
        segment = audio[start:start + segment_length]
        segments.append(segment)
    
    return segments


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range.
    
    Args:
        audio: Audio signal
        
    Returns:
        Normalized audio
    """
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio / max_val
    return audio


def augment_audio(
    audio: np.ndarray,
    sample_rate: int = 44100,
    pitch_shift: Optional[float] = None,
    time_stretch: Optional[float] = None,
    noise_level: float = 0.0
) -> np.ndarray:
    """Apply data augmentation to audio.
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate
        pitch_shift: Pitch shift in semitones (None = no shift)
        time_stretch: Time stretch factor (None = no stretch)
        noise_level: Noise level (0.0 = no noise)
        
    Returns:
        Augmented audio
    """
    result = audio.copy()
    
    # Pitch shift
    if pitch_shift is not None and pitch_shift != 0:
        result = librosa.effects.pitch_shift(
            result, sr=sample_rate, n_steps=pitch_shift
        )
    
    # Time stretch
    if time_stretch is not None and time_stretch != 1.0:
        result = librosa.effects.time_stretch(result, rate=time_stretch)
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, len(result))
        result = result + noise
    
    return result

