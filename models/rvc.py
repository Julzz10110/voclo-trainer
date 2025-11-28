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
        
        # Mel to audio conversion (inverse mel projection)
        # This is the inverse of audio_to_mel for converting mel back to magnitude
        self.mel_to_audio = nn.Linear(n_mels, n_freq_bins, bias=False)
        # Initialize as pseudo-inverse (transpose of audio_to_mel)
        with torch.no_grad():
            # Use transpose as approximate inverse for simple averaging projection
            self.mel_to_audio.weight.copy_(self.audio_to_mel.weight.t())
    
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
        Converts audio to mel spectrogram, processes it, and returns converted raw audio.
        
        Args:
            audio: Input audio [batch, time] - raw audio samples
            speaker_id: Speaker ID [batch] (optional, defaults to 0)
            
        Returns:
            Converted raw audio [batch, time]
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
        # ONNX doesn't support complex types, so we use return_complex=False
        # which returns [..., 2] tensor directly (real, imag) without complex type
        if batch_size == 1:
            # Use return_complex=False to avoid complex types in ONNX export
            # Returns [n_fft//2 + 1, time_frames, 2] where last dim is [real, imag]
            stft_real = torch.stft(
                audio.squeeze(0),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=torch.hann_window(self.n_fft, device=audio.device),
                center=True,
                normalized=False,
                onesided=True,
                return_complex=False  # Returns [..., 2] directly, no complex type
            )  # [n_fft//2 + 1, time_frames, 2]
            # Save phase for ISTFT later
            real_part = stft_real[..., 0]  # [n_fft//2 + 1, time_frames]
            imag_part = stft_real[..., 1]  # [n_fft//2 + 1, time_frames]
            magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)  # [n_fft//2 + 1, time_frames]
            # Compute phase: atan2(imag, real)
            phase = torch.atan2(imag_part, real_part)  # [n_fft//2 + 1, time_frames]
            magnitude = magnitude.unsqueeze(0)  # [1, n_fft//2 + 1, time_frames]
            phase = phase.unsqueeze(0)  # [1, n_fft//2 + 1, time_frames]
        else:
            # For batch > 1, process each item (may not export well to ONNX)
            # In practice, voclo processes one sample at a time
            stft_results = []
            phase_results = []
            for i in range(batch_size):
                # Use return_complex=False to avoid complex types in ONNX export
                stft_real = torch.stft(
                    audio[i],
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.n_fft,
                    window=torch.hann_window(self.n_fft, device=audio.device),
                    center=True,
                    normalized=False,
                    onesided=True,
                    return_complex=False  # Returns [..., 2] directly, no complex type
                )  # [n_fft//2 + 1, time_frames, 2]
                # Compute magnitude and phase
                real_part = stft_real[..., 0]
                imag_part = stft_real[..., 1]
                magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
                phase = torch.atan2(imag_part, real_part)
                stft_results.append(magnitude)
                phase_results.append(phase)
            magnitude = torch.stack(stft_results, dim=0)  # [batch, n_fft//2 + 1, time_frames]
            phase = torch.stack(phase_results, dim=0)  # [batch, n_fft//2 + 1, time_frames]
        
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
        output_mel = output["mel"]  # [batch, n_mels, time_frames]
        
        # Convert mel back to magnitude
        # Denormalize and inverse log
        output_mel = output_mel * (output_mel.std(dim=-1, keepdim=True) + 1e-10) + output_mel.mean(dim=-1, keepdim=True)
        output_mel = torch.exp(output_mel) - 1e-10
        
        # Reshape for linear layer: [batch, time_frames, n_mels]
        output_mel = output_mel.transpose(1, 2)
        
        # Apply inverse mel projection: [batch, time_frames, n_fft//2 + 1]
        output_magnitude = self.mel_to_audio(output_mel)
        
        # Transpose back to [batch, n_fft//2 + 1, time_frames]
        output_magnitude = output_magnitude.transpose(1, 2)
        
        # For ONNX compatibility, we need to avoid view_as_complex and istft
        # Since these are not supported, we'll use a workaround:
        # Return the magnitude spectrum and let the post-processing handle conversion
        # OR use a simplified vocoder that works with ONNX
        
        # Actually, the best approach is to use torch.istft directly with real/imag
        # But since view_as_complex doesn't work, we need to use a custom ISTFT
        # implementation using only ONNX-compatible operations.
        
        # For now, let's use a simplified approach: use torch.istft with a workaround
        # We'll manually implement ISTFT using only basic operations
        
        # Calculate output length from STFT parameters
        time_frames = output_magnitude.size(-1)
        original_length = audio.size(-1)
        expected_length = (time_frames - 1) * self.hop_length + self.n_fft
        
        # Reconstruct STFT using original phase
        # Convert magnitude and phase back to real/imag
        output_real = output_magnitude * torch.cos(phase)  # [batch, n_fft//2 + 1, time_frames]
        output_imag = output_magnitude * torch.sin(phase)  # [batch, n_fft//2 + 1, time_frames]
        
        # For ONNX compatibility, we need to avoid all FFT/IFFT operations
        # Since torch.istft, torch.view_as_complex, and torch.fft.irfft are not supported,
        # we'll use a very simplified vocoder that uses only basic ONNX operations
        
        # Simplified vocoder approach:
        # 1. Use magnitude spectrum to estimate audio amplitude
        # 2. Use phase to create a simplified waveform
        # 3. Use overlap-add with basic operations only
        
        # Calculate output length
        output_length = min(original_length, expected_length)
        
        # Simplified reconstruction: use magnitude to create audio envelope
        # and interpolate to match original length
        if batch_size == 1:
            # Get mean magnitude across frequency bins for each time frame
            # This gives us a rough estimate of audio amplitude over time
            mean_magnitude = output_magnitude.squeeze(0).mean(dim=0)  # [time_frames]
            
            # Interpolate to match original audio length
            if mean_magnitude.size(-1) > 1:
                # Use linear interpolation (ONNX-compatible)
                indices = torch.linspace(0, mean_magnitude.size(-1) - 1, output_length, device=audio.device)
                indices_floor = indices.floor().long()
                indices_ceil = (indices_floor + 1).clamp(0, mean_magnitude.size(-1) - 1)
                alpha = indices - indices_floor.float()
                
                # Linear interpolation
                audio_envelope = mean_magnitude[indices_floor] * (1 - alpha) + mean_magnitude[indices_ceil] * alpha
            else:
                audio_envelope = mean_magnitude.repeat(output_length // mean_magnitude.size(-1) + 1)[:output_length]
            
            # Create simplified audio signal using the envelope
            # We'll use a combination of sine waves at different frequencies
            # weighted by the magnitude spectrum
            
            # Get dominant frequencies from magnitude spectrum
            # Use the frequency bin with maximum magnitude for each frame
            max_mag_indices = output_magnitude.squeeze(0).argmax(dim=0)  # [time_frames]
            
            # Convert bin indices to frequencies
            freq_bins = max_mag_indices.float() * (self.sample_rate / 2) / (self.n_fft // 2 + 1)
            
            # Interpolate frequencies to match output length
            if freq_bins.size(-1) > 1:
                indices = torch.linspace(0, freq_bins.size(-1) - 1, output_length, device=audio.device)
                indices_floor = indices.floor().long()
                indices_ceil = (indices_floor + 1).clamp(0, freq_bins.size(-1) - 1)
                alpha = indices - indices_floor.float()
                freq_interp = freq_bins[indices_floor] * (1 - alpha) + freq_bins[indices_ceil] * alpha
            else:
                freq_interp = freq_bins.repeat(output_length // freq_bins.size(-1) + 1)[:output_length]
            
            # Create time axis
            time_axis = torch.arange(output_length, dtype=torch.float32, device=audio.device) / self.sample_rate
            
            # Generate simplified audio using envelope and frequency
            # This is a very simplified vocoder but ONNX-compatible
            output_audio_1d = audio_envelope * torch.sin(2 * torch.pi * freq_interp * time_axis)
            
            # Normalize to match input scale
            if output_audio_1d.abs().max() > 0:
                output_audio_1d = output_audio_1d / (output_audio_1d.abs().max() + 1e-10)
                if audio.abs().max() > 0:
                    output_audio_1d = output_audio_1d * audio.abs().max()
            
            output_audio = output_audio_1d.unsqueeze(0)  # [1, time]
        else:
            # For batch > 1, process each item
            audio_results = []
            for i in range(batch_size):
                mean_mag = output_magnitude[i].mean(dim=0)  # [time_frames]
                output_length = min(original_length, expected_length)
                
                if mean_mag.size(-1) > 1:
                    indices = torch.linspace(0, mean_mag.size(-1) - 1, output_length, device=audio.device)
                    indices_floor = indices.floor().long()
                    indices_ceil = (indices_floor + 1).clamp(0, mean_mag.size(-1) - 1)
                    alpha = indices - indices_floor.float()
                    audio_envelope = mean_mag[indices_floor] * (1 - alpha) + mean_mag[indices_ceil] * alpha
                else:
                    audio_envelope = mean_mag.repeat(output_length // mean_mag.size(-1) + 1)[:output_length]
                
                max_mag_indices = output_magnitude[i].argmax(dim=0)
                freq_bins = max_mag_indices.float() * (self.sample_rate / 2) / (self.n_fft // 2 + 1)
                
                if freq_bins.size(-1) > 1:
                    indices = torch.linspace(0, freq_bins.size(-1) - 1, output_length, device=audio.device)
                    indices_floor = indices.floor().long()
                    indices_ceil = (indices_floor + 1).clamp(0, freq_bins.size(-1) - 1)
                    alpha = indices - indices_floor.float()
                    freq_interp = freq_bins[indices_floor] * (1 - alpha) + freq_bins[indices_ceil] * alpha
                else:
                    freq_interp = freq_bins.repeat(output_length // freq_bins.size(-1) + 1)[:output_length]
                
                time_axis = torch.arange(output_length, dtype=torch.float32, device=audio.device) / self.sample_rate
                audio_1d = audio_envelope * torch.sin(2 * torch.pi * freq_interp * time_axis)
                
                if audio_1d.abs().max() > 0:
                    audio_1d = audio_1d / (audio_1d.abs().max() + 1e-10)
                    if audio[i].abs().max() > 0:
                        audio_1d = audio_1d * audio[i].abs().max()
                
                audio_results.append(audio_1d)
            output_audio = torch.stack(audio_results, dim=0)  # [batch, time]
        
        return output_audio

