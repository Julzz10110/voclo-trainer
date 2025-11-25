"""Base class for voice conversion models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional


class BaseVoiceConversionModel(nn.Module, ABC):
    """Base class for voice conversion models."""
    
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        n_fft: int = 2048,
        n_mels: int = 80
    ):
        """Initialize base model.
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Hop length for spectrogram
            n_fft: FFT window size
            n_mels: Number of mel bins
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
    
    @abstractmethod
    def forward(
        self,
        audio: torch.Tensor,
        speaker_id: Optional[torch.Tensor] = None,
        f0: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            audio: Input audio [batch, time]
            speaker_id: Speaker ID [batch] (optional)
            f0: F0 values [batch, time] (optional)
            
        Returns:
            Dictionary with model outputs
        """
        pass
    
    @abstractmethod
    def inference(
        self,
        audio: torch.Tensor,
        speaker_id: Optional[torch.Tensor] = None,
        f0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Inference mode (no gradients).
        
        Args:
            audio: Input audio [batch, time]
            speaker_id: Speaker ID [batch] (optional)
            f0: F0 values [batch, time] (optional)
            
        Returns:
            Converted audio [batch, time]
        """
        pass
    
    def get_model_info(self) -> Dict:
        """Get model information for export.
        
        Returns:
            Dictionary with model parameters
        """
        return {
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "n_fft": self.n_fft,
            "n_mels": self.n_mels
        }

