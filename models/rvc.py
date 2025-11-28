"""RVC (Retrieval-based Voice Conversion) model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .base import BaseVoiceConversionModel


class ResidualBlock(nn.Module):
    """Residual block with dilation."""
    
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3,
            dilation=dilation, padding=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3,
            dilation=dilation, padding=dilation
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)


class RVCEncoder(nn.Module):
    """Encoder for RVC model."""
    
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()
        self.input_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dilation=2**i)
            for i in range(num_layers)
        ])
        
        self.output_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, mel_bins, time]
        x = self.input_conv(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.output_conv(x)
        return x


class RVCDecoder(nn.Module):
    """Decoder for RVC model."""
    
    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 80,
        num_layers: int = 3
    ):
        super().__init__()
        self.input_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dilation=2**i)
            for i in range(num_layers)
        ])
        
        self.output_conv = nn.Conv1d(hidden_dim, output_dim, kernel_size=7, padding=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, hidden_dim, time]
        x = self.input_conv(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.output_conv(x)
        return x


class SpeakerEmbedding(nn.Module):
    """Speaker embedding layer."""
    
    def __init__(self, num_speakers: int = 1, embed_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_speakers, embed_dim)
    
    def forward(self, speaker_id: torch.Tensor) -> torch.Tensor:
        # speaker_id: [batch]
        return self.embedding(speaker_id)  # [batch, embed_dim]


class RVCModel(BaseVoiceConversionModel):
    """RVC (Retrieval-based Voice Conversion) model.
    
    This is a simplified RVC implementation suitable for training
    and export to ONNX.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        n_fft: int = 2048,
        n_mels: int = 80,
        hidden_size: int = 256,
        num_layers: int = 3,
        num_speakers: int = 1,
        embed_dim: int = 256,
        f0_bin: int = 256,
        f0_min: float = 50.0,
        f0_max: float = 1100.0
    ):
        """Initialize RVC model.
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Hop length for spectrogram
            n_fft: FFT window size
            n_mels: Number of mel bins
            hidden_size: Hidden dimension
            num_layers: Number of residual layers
            num_speakers: Number of speakers
            embed_dim: Speaker embedding dimension
            f0_bin: F0 quantization bins
            f0_min: Minimum F0
            f0_max: Maximum F0
        """
        super().__init__(sample_rate, hop_length, n_fft, n_mels)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_speakers = num_speakers
        self.embed_dim = embed_dim
        self.f0_bin = f0_bin
        self.f0_min = f0_min
        self.f0_max = f0_max
        
        # Components
        self.encoder = RVCEncoder(n_mels, hidden_size, num_layers)
        self.decoder = RVCDecoder(hidden_size, n_mels, num_layers)
        self.speaker_embedding = SpeakerEmbedding(num_speakers, embed_dim)
        
        # F0 processing
        self.f0_embedding = nn.Linear(1, hidden_size)
        
        # Audio to mel conversion for ONNX export
        # Create mel projection layer (simplified mel filter bank)
        n_freq_bins = n_fft // 2 + 1
        self.audio_to_mel = nn.Linear(n_freq_bins, n_mels, bias=False)
        # Initialize with simple averaging projection
        with torch.no_grad():
            weight = torch.zeros(n_mels, n_freq_bins)
            bins_per_mel = n_freq_bins / n_mels
            for i in range(n_mels):
                start = int(i * bins_per_mel)
                end = int((i + 1) * bins_per_mel)
                if end > start:
                    weight[i, start:end] = 1.0 / (end - start)
            self.audio_to_mel.weight.copy_(weight)
    
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
            Dictionary with:
                - mel: Reconstructed mel spectrogram
                - encoded: Encoded features
        """
        # For training, we expect mel spectrogram as input [batch, n_mels, time]
        # When called from forward_audio, mel is already in correct format [batch, n_mels, time_frames]
        # When called during training, audio might be mel [batch, n_mels, time] or raw audio [batch, time]
        if audio.dim() == 3:
            # Already in correct format [batch, n_mels, time]
            mel = audio
        elif audio.dim() == 2:
            # Could be mel [batch, n_mels] or raw audio [batch, time]
            # Check if second dimension matches n_mels
            if audio.size(1) == self.n_mels:
                # This is mel in wrong format [batch, n_mels] - need to add time dimension
                # This shouldn't happen in normal flow, but handle it
                mel = audio.unsqueeze(-1)  # [batch, n_mels, 1]
            else:
                # This is raw audio [batch, time] - should not happen in forward
                # This means forward was called incorrectly
                raise ValueError(
                    f"forward() received raw audio [batch={audio.size(0)}, time={audio.size(1)}], "
                    f"but expects mel spectrogram [batch, n_mels={self.n_mels}, time]. "
                    f"Use forward_audio() for raw audio input."
                )
        else:
            raise ValueError(f"Unexpected input dimension: {audio.dim()}, expected 2 or 3")
        
        # Encode
        encoded = self.encoder(mel)
        
        # Add speaker embedding if provided
        if speaker_id is not None:
            speaker_emb = self.speaker_embedding(speaker_id)
            # Broadcast to match time dimension
            speaker_emb = speaker_emb.unsqueeze(-1)  # [batch, embed_dim, 1]
            speaker_emb = speaker_emb.expand(-1, -1, encoded.size(-1))
            encoded = encoded + speaker_emb
        
        # Add F0 information if provided
        if f0 is not None:
            # Normalize F0
            f0_normalized = (f0 - self.f0_min) / (self.f0_max - self.f0_min)
            f0_normalized = f0_normalized.clamp(0, 1).unsqueeze(1)  # [batch, 1, time]
            f0_emb = self.f0_embedding(f0_normalized.transpose(1, 2))  # [batch, time, hidden]
            f0_emb = f0_emb.transpose(1, 2)  # [batch, hidden, time]
            encoded = encoded + f0_emb
        
        # Decode
        mel_recon = self.decoder(encoded)
        
        return {
            "mel": mel_recon,
            "encoded": encoded
        }
    
    def inference(
        self,
        audio: torch.Tensor,
        speaker_id: Optional[torch.Tensor] = None,
        f0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Inference mode - convert audio.
        
        Args:
            audio: Input audio [batch, time] or mel [batch, n_mels, time]
            speaker_id: Speaker ID [batch] (optional)
            f0: F0 values [batch, time] (optional)
            
        Returns:
            Converted mel spectrogram [batch, n_mels, time]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(audio, speaker_id, f0)
            return output["mel"]
    
    def forward_audio(
        self,
        audio: torch.Tensor,
        speaker_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with raw audio input.
        
        This is the interface expected by ONNX export.
        Converts audio to mel spectrogram, processes it, and returns converted mel.
        
        Args:
            audio: Input audio [batch, time] - raw audio samples
            speaker_id: Speaker ID [batch] (optional, defaults to 0)
            
        Returns:
            Converted mel spectrogram [batch, n_mels, time_frames]
        """
        if speaker_id is None:
            speaker_id = torch.zeros(audio.size(0), dtype=torch.long, device=audio.device)
        
        # Convert audio to mel spectrogram using torch STFT
        # This is ONNX-compatible
        batch_size = audio.size(0)
        
        # Ensure audio is 2D [batch, time]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Compute STFT - process first sample (ONNX doesn't handle loops well)
        # For batch processing, we'll handle one at a time or use batch STFT if available
        # For ONNX compatibility, we process batch_size=1 case
        # ONNX doesn't support complex types, so we use return_complex=True
        # and convert to real representation using view_as_real
        if batch_size == 1:
            stft = torch.stft(
                audio.squeeze(0),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=torch.hann_window(self.n_fft, device=audio.device),
                center=True,
                normalized=False,
                onesided=True,
                return_complex=True
            )
            # Convert complex to real representation for ONNX compatibility
            # view_as_real converts [..., complex] to [..., 2] where last dim is [real, imag]
            stft_real = torch.view_as_real(stft)  # [n_fft//2 + 1, time_frames, 2]
            # Compute magnitude: sqrt(real^2 + imag^2)
            real_part = stft_real[..., 0]  # [n_fft//2 + 1, time_frames]
            imag_part = stft_real[..., 1]  # [n_fft//2 + 1, time_frames]
            magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2).unsqueeze(0)  # [1, n_fft//2 + 1, time_frames]
        else:
            # For batch > 1, process each item (may not export well to ONNX)
            # In practice, voclo processes one sample at a time
            stft_results = []
            for i in range(batch_size):
                stft = torch.stft(
                    audio[i],
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.n_fft,
                    window=torch.hann_window(self.n_fft, device=audio.device),
                    center=True,
                    normalized=False,
                    onesided=True,
                    return_complex=True
                )
                # Convert complex to real representation for ONNX compatibility
                stft_real = torch.view_as_real(stft)
                real_part = stft_real[..., 0]
                imag_part = stft_real[..., 1]
                magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
                stft_results.append(magnitude)
            magnitude = torch.stack(stft_results, dim=0)  # [batch, n_fft//2 + 1, time_frames]
        
        # Reshape for linear layer: [batch, time_frames, n_fft//2 + 1]
        magnitude = magnitude.transpose(1, 2)
        
        # Apply mel projection: [batch, time_frames, n_mels]
        mel = self.audio_to_mel(magnitude)
        
        # Transpose back to [batch, n_mels, time_frames]
        mel = mel.transpose(1, 2)
        
        # Log scale (add small epsilon to avoid log(0))
        mel = torch.log(mel + 1e-10)
        
        # Normalize (optional, but helps with training stability)
        mel = (mel - mel.mean(dim=-1, keepdim=True)) / (mel.std(dim=-1, keepdim=True) + 1e-10)
        
        # Process through model
        output = self.forward(mel, speaker_id, None)
        return output["mel"]

