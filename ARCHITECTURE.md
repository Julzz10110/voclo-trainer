# Voclo Trainer Architecture

This document describes the architecture of the voclo-trainer training system.

## Overview

Voclo-trainer consists of the following main components:

1. **Models** (`models/`) - Neural network architectures
2. **Utilities** (`utils/`) - Helper functions
3. **Training Scripts** - Main scripts for training and export
4. **Configuration** (`configs/`) - Configuration files

## Project Structure

```
voclo-trainer/
├── models/              # Model architectures
│   ├── base.py         # Base class for models
│   ├── rvc.py          # RVC model
│   └── __init__.py
├── utils/               # Utilities
│   ├── audio.py        # Audio processing
│   ├── dataset.py      # Dataset classes
│   └── __init__.py
├── configs/             # Configurations
│   └── default_config.yaml
├── train.py            # Training script
├── export_onnx.py      # ONNX export
├── preprocess.py       # Data preprocessing
├── test_model.py       # Model testing
└── requirements.txt    # Dependencies
```

## Models

### BaseVoiceConversionModel

Base class for all voice conversion models. Defines interface:
- `forward()` - Forward pass for training
- `inference()` - Inference without gradients
- `forward_audio()` - Interface for ONNX export

### RVCModel

Simplified implementation of RVC (Retrieval-based Voice Conversion):
- **Encoder**: Converts mel spectrogram to hidden representation
- **Decoder**: Reconstructs mel spectrogram from hidden representation
- **Speaker Embedding**: Speaker information embedding
- **F0 Processing**: Pitch information processing

Architecture:
```
Input (mel) → Encoder → [Speaker Embedding] → [F0 Embedding] → Decoder → Output (mel)
```

## Data Processing

### VoiceConversionDataset

Dataset class for training:
- Loads audio files from structured directories
- Extracts mel spectrograms and F0
- Applies data augmentation
- Returns batches for training

### Audio Utilities

Functions for audio processing:
- `load_audio()` - Loading and resampling
- `extract_mel_spectrogram()` - Extract mel spectrogram
- `extract_f0()` - Extract F0 (fundamental frequency)
- `augment_audio()` - Data augmentation

## Training Process

1. **Data Loading**: Dataset loads and preprocesses audio
2. **Forward pass**: Model processes mel spectrograms
3. **Loss computation**: Loss is computed between prediction and target mel
4. **Backward pass**: Gradients are computed and applied
5. **Validation**: Periodic validation on separate dataset
6. **Checkpointing**: Saving best models

## ONNX Export

Export process:
1. Load trained model from checkpoint
2. Create dummy input
3. Export via `torch.onnx.export()`
4. Simplify model (optional)
5. Validate ONNX model
6. Save with correct name

### ONNX Model Requirements

For compatibility with voclo:
- Input: `audio` [batch, time] - Float32
- Output: `output` [batch, time] - Float32
- Dynamic axes for batch and time
- Opset version 14+

## Integration with voclo

Voclo expects models in format:
- File: `{name}_sr{sample_rate}_hop{hop_length}_nfft{n_fft}.onnx`
- Parameters are extracted from filename
- Model should accept audio and return audio

## Extending the System

### Adding New Architecture

1. Create a class inheriting from `BaseVoiceConversionModel`
2. Implement `forward()`, `inference()`, `forward_audio()`
3. Add support in `train.py` and `export_onnx.py`
4. Update configuration

### Adding New Utilities

Add functions to appropriate modules in `utils/`:
- `audio.py` - for audio processing
- `dataset.py` - for data handling

## Performance

### Optimizations

- Using GPU for training
- Batch processing
- Data loading with multiple workers
- Pin memory for fast transfer to GPU
- Gradient accumulation for large models

### Recommendations

- Use `num_workers > 0` for data loading
- Enable `pin_memory=True` when using GPU
- Use `gradient_clip` for training stability
- Save checkpoints regularly

## Future Improvements

- Support for other architectures (So-VITS-SVC, VITS)
- Vocoder integration for end-to-end training
- Multi-speaker model support
- Automatic hyperparameter optimization
- Web UI for training
