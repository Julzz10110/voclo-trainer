"""Export trained model to ONNX format for voclo."""

import argparse
import torch
import torch.nn
import torch.onnx
import yaml
from pathlib import Path
import logging
import onnx

# Optional onnxsim import (only needed to simplify the model)
try:
    import onnxsim
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False

from models.rvc import RVCModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device = torch.device('cpu')
) -> tuple:
    """Load model from checkpoint.
    
    Returns:
        Tuple of (model, config)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    if not config:
        raise ValueError("Checkpoint does not contain configuration")
    
    # Create model
    model_config = config['model']
    if model_config['name'] == 'rvc':
        model = RVCModel(
            sample_rate=model_config['sample_rate'],
            hop_length=model_config['hop_length'],
            n_fft=model_config['n_fft'],
            n_mels=model_config['n_mels'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config.get('num_layers', 3),
            num_speakers=model_config.get('num_speakers', 1),
            embed_dim=model_config.get('embed_dim', 256),
            f0_bin=model_config.get('f0_bin', 256),
            f0_min=model_config.get('f0_min', 50.0),
            f0_max=model_config.get('f0_max', 1100.0)
        )
    else:
        raise ValueError(f"Unknown model type: {model_config['name']}")
    
    # Load weights - allow missing keys for backward compatibility
    # (e.g., audio_to_mel layer added in newer versions)
    state_dict = checkpoint['model_state_dict']
    model_state = model.state_dict()
    
    # Filter out keys that don't exist in the model
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state}
    missing_keys = set(model_state.keys()) - set(state_dict.keys())
    unexpected_keys = set(state_dict.keys()) - set(model_state.keys())
    
    if missing_keys:
        logger.warning(f"Missing keys in checkpoint (will use default initialization): {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint (will be ignored): {unexpected_keys}")
    
    # Load available weights
    model.load_state_dict(filtered_state_dict, strict=False)
    
    # If audio_to_mel is missing, it will be initialized with default weights from __init__
    # This is fine for ONNX export as the layer is used for audio-to-mel conversion
    model.eval()
    
    logger.info("Model loaded successfully")
    return model, config


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    sample_rate: int = 44100,
    hop_length: int = 512,
    n_fft: int = 2048,
    opset_version: int = 17,  # Minimum version for STFT support
    simplify: bool = True
):
    """Export model to ONNX format.
    
    Args:
        model: Trained model
        output_path: Output ONNX file path
        sample_rate: Sample rate
        hop_length: Hop length
        n_fft: FFT window size
        opset_version: ONNX opset version
        simplify: Whether to simplify the ONNX model
    """
    logger.info("Exporting model to ONNX...")
    
    # Create dummy input - raw audio format
    # Model expects raw audio [batch, time] - voclo will pass this format
    # Using a typical buffer size (~100ms at 44.1kHz = 4410 samples)
    dummy_audio_length = 4410
    dummy_audio = torch.randn(1, dummy_audio_length)
    
    # Create wrapper class for ONNX export
    # ONNX export needs a module, not just a function
    class ModelWrapper(torch.nn.Module):
        """Wrapper module that uses forward_audio for ONNX export."""
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        
        def forward(self, audio: torch.Tensor):
            """Forward pass using forward_audio method."""
            return self.base_model.forward_audio(audio, None)
    
    # Create wrapped model
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    # Export
    logger.info(f"Exporting with input shape: {dummy_audio.shape} (raw audio)")
    
    try:
        # Use the old export method (dynamo=False) for compatibility
        # The new torch.export may have issues with some models
        torch.onnx.export(
            wrapped_model,
            (dummy_audio,),
            output_path,
            input_names=['audio'],
            output_names=['output'],
            dynamic_axes={
                'audio': {0: 'batch', 1: 'time'},
                'output': {0: 'batch', 1: 'time'}  # Model now returns raw audio, not mel
            },
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
            dynamo=False  # Use old export method for compatibility
        )
        logger.info(f"ONNX model exported to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export ONNX model: {e}")
        raise
    
    # Simplify model if requested
    if simplify:
        if not ONNXSIM_AVAILABLE:
            logger.warning("onnxsim не установлен. Пропускаем упрощение модели.")
            logger.info("Модель экспортирована без упрощения. Это нормально, модель будет работать.")
        else:
            try:
                logger.info("Simplifying ONNX model...")
                onnx_model = onnx.load(output_path)
                simplified_model, check = onnxsim.simplify(onnx_model)
                if check:
                    onnx.save(simplified_model, output_path)
                    logger.info("ONNX model simplified successfully")
                else:
                    logger.warning("ONNX model simplification check failed, using original")
            except Exception as e:
                logger.warning(f"Failed to simplify ONNX model: {e}")
    
    # Verify ONNX model
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed")
    except Exception as e:
        logger.error(f"ONNX model verification failed: {e}")
        raise


def generate_model_name(
    base_name: str,
    sample_rate: int,
    hop_length: int,
    n_fft: int
) -> str:
    """Generate model filename following voclo convention.
    
    Format: {name}_sr{sample_rate}_hop{hop_length}_nfft{n_fft}.onnx
    """
    return f"{base_name}_sr{sample_rate}_hop{hop_length}_nfft{n_fft}.onnx"


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output ONNX file path (default: auto-generated from model name)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Model name (for auto-generating filename)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=None,
        help='Sample rate (overrides config)'
    )
    parser.add_argument(
        '--hop-length',
        type=int,
        default=None,
        help='Hop length (overrides config)'
    )
    parser.add_argument(
        '--n-fft',
        type=int,
        default=None,
        help='N_FFT (overrides config)'
    )
    parser.add_argument(
        '--opset-version',
        type=int,
        default=17,  # Minimum version for STFT support
        help='ONNX opset version (minimum 17 for STFT support)'
    )
    parser.add_argument(
        '--no-simplify',
        action='store_true',
        help='Disable ONNX model simplification'
    )
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cpu')  # ONNX export should be on CPU
    
    # Load model
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    
    # Get model parameters
    model_config = config['model']
    sample_rate = args.sample_rate or model_config['sample_rate']
    hop_length = args.hop_length or model_config['hop_length']
    n_fft = args.n_fft or model_config['n_fft']
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Auto-generate filename
        if args.name:
            model_name = args.name
        else:
            # Extract from checkpoint filename
            checkpoint_name = Path(args.checkpoint).stem
            model_name = checkpoint_name.replace('model_epoch_', '').replace('best_', '')
        
        output_path = generate_model_name(model_name, sample_rate, hop_length, n_fft)
        
        # Try to save to voclo/models directory
        voclo_models_dir = Path('../voclo/models')
        if voclo_models_dir.exists():
            output_path = str(voclo_models_dir / output_path)
            logger.info(f"Output directory: {voclo_models_dir}")
        else:
            logger.warning(f"voclo/models directory not found, saving to current directory")
    
    # Export
    export_to_onnx(
        model=model,
        output_path=output_path,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
        opset_version=args.opset_version,
        simplify=not args.no_simplify
    )
    
    logger.info(f"\n✅ Model exported successfully!")
    logger.info(f"   Output: {output_path}")
    logger.info(f"   Sample rate: {sample_rate}")
    logger.info(f"   Hop length: {hop_length}")
    logger.info(f"   N_FFT: {n_fft}")
    logger.info(f"\nTo use in voclo, copy the model to voclo/models/ directory")


if __name__ == '__main__':
    main()




