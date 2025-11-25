# Quick Start - Voclo Trainer

This quick guide will help you get started training models for voclo.

## Step 1: Installation

```bash
# Navigate to voclo-trainer directory
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

## Step 2: Prepare Data

### Data Structure

Create the following directory structure:

```
data/
  train/
    speaker1/          # Or character name
      audio1.wav
      audio2.wav
      ...
  val/
    speaker1/
      audio1.wav
      ...
```

### Audio Preprocessing (Optional)

```bash
python preprocess.py \
  --input raw_data/ \
  --output data/train/ \
  --sample-rate 44100
```

## Step 3: Configure

Edit `configs/default_config.yaml`:

```yaml
model:
  name: "rvc"
  sample_rate: 44100
  hop_length: 512
  n_fft: 2048

data:
  train_dir: "data/train"
  val_dir: "data/val"

training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.0001
```

## Step 4: Training

```bash
python train.py --config configs/default_config.yaml
```

Training may take several hours or days depending on:
- Dataset size
- Number of epochs
- GPU power

Checkpoints are saved in `checkpoints/`:
- `best_model.pth` - best model by validation
- `model_epoch_N.pth` - checkpoints every N epochs

## Step 5: Export to ONNX

After training, export the model:

```bash
python export_onnx.py \
  --checkpoint checkpoints/best_model.pth \
  --name my_character \
  --output ../voclo/models/my_character_sr44100_hop512_nfft2048.onnx
```

Or use automatic naming:

```bash
python export_onnx.py \
  --checkpoint checkpoints/best_model.pth \
  --name anime_character_01
```

The model will be automatically saved in the correct format.

## Step 6: Using in voclo

1. Copy the `.onnx` file to `voclo/models/`
2. Launch voclo
3. The model will be automatically detected
4. Select the model in the voclo interface

## Tips

### Data Quality
- Use clean recordings without background noise
- Minimum 10-30 minutes of audio for good quality
- Variety of phrases and intonations
- Same sample rate (44100 Hz recommended)

### Training
- Start with a small number of epochs (50-100)
- Monitor validation loss
- Use GPU for acceleration
- Save checkpoints regularly

### Export
- Always export from `best_model.pth`
- Check parameters (sample_rate, hop_length, n_fft)
- Make sure the model works in voclo before long training

## Troubleshooting

### Error "No audio files found"
- Check paths in configuration
- Make sure files have extensions .wav, .mp3, .flac

### CUDA out of memory error
- Reduce `batch_size` in configuration
- Use smaller `segment_length`

### Model doesn't work in voclo
- Check model parameters (sample_rate, hop_length, n_fft)
- Make sure the model is exported correctly
- Check voclo logs for errors

## Additional Information

See `README.md` for detailed documentation.
