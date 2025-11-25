# Voclo Trainer

Training system for creating ONNX models for speech-to-speech voice conversion.

## Features

- 🎤 Train voice conversion models on your audio data
- 🔄 Support for various architectures (RVC, So-VITS-SVC, VITS)
- 📦 Export trained models to ONNX format for use in voclo
- 🎯 Automatic model naming with parameters (sample_rate, hop_length, n_fft)
- 🛠️ Data preparation and audio preprocessing utilities

## Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- PyTorch 2.0+
- ONNX Runtime

## Installation

```bash
# Clone the repository
cd voclo-trainer

# Create a virtual environment
python -m venv venv

# Activate the environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

Place your audio files in the following structure:
```
data/
  train/
    speaker1/
      audio1.wav
      audio2.wav
      ...
    speaker2/
      audio1.wav
      ...
  val/
    speaker1/
      audio1.wav
      ...
```

### 2. Train Model

```bash
python train.py --config configs/default_config.yaml
```

### 3. Export to ONNX

```bash
python export_onnx.py --checkpoint checkpoints/model_epoch_100.pth --output ../voclo/models/my_character_sr44100_hop512_nfft2048.onnx
```

## Project Structure

```
voclo-trainer/
├── configs/              # Configuration files
├── data/                 # Training data
├── models/               # Model architectures
├── utils/                # Utilities
├── train.py             # Main training script
├── export_onnx.py       # ONNX export
├── preprocess.py        # Data preprocessing
└── requirements.txt     # Dependencies
```

## Configuration

Create a configuration file in `configs/`:

```yaml
model:
  name: "rvc"
  sample_rate: 44100
  hop_length: 512
  n_fft: 2048
  n_mels: 80
  hidden_size: 256

training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.0001
  save_interval: 10

data:
  train_dir: "data/train"
  val_dir: "data/val"
  segment_length: 8192
```

## Model Export

After training, export the model to ONNX:

```bash
python export_onnx.py \
  --checkpoint checkpoints/best_model.pth \
  --output ../voclo/models/character_name_sr44100_hop512_nfft2048.onnx \
  --sample_rate 44100 \
  --hop_length 512 \
  --n_fft 2048
```

The model will be automatically saved in a format compatible with voclo.

## Using in voclo

After export, simply copy the `.onnx` file to the `voclo/models/` folder. Voclo will automatically detect the model on the next launch.

### Testing the Model

Before using in voclo, it's recommended to test the model:

```bash
python test_model.py --model ../voclo/models/my_model.onnx --audio test.wav --output output.wav
```

## Additional Documentation

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [EXPORT_GUIDE.md](EXPORT_GUIDE.md) - Detailed export guide

## Supported Architectures

- **RVC (Retrieval-based Voice Conversion)** - popular for anime voices
- **So-VITS-SVC** - high quality, requires more data
- **VITS** - fast inference, good quality

## Training in Google Colab

To train in Google Colab, use the provided notebook:

1. Open `voclo_trainer_colab.ipynb` in Google Colab
2. Follow the instructions in the notebook
3. See [COLAB_GUIDE.md](COLAB_GUIDE.md) for detailed guide

## License

See LICENSE file in the project root.
