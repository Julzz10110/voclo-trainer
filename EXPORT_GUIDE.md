# Model Export Guide

This guide explains how to properly export trained models to ONNX format for use in voclo.

## Important Parameters

When exporting a model, make sure the following parameters match:

- **sample_rate**: 44100 (recommended) or 48000
- **hop_length**: 512 (standard value)
- **n_fft**: 2048 (standard value)

These parameters must match the parameters used during training.

## Filename Format

Voclo automatically determines model parameters from the filename:

```
{name}_sr{sample_rate}_hop{hop_length}_nfft{n_fft}.onnx
```

Examples:
- `anime_character_01_sr44100_hop512_nfft2048.onnx`
- `my_voice_sr48000_hop512_nfft2048.onnx`

If parameters are not specified in the name, voclo uses default values:
- sample_rate: 44100
- hop_length: 512
- n_fft: 2048

## Export Process

### 1. Train Model

First, train the model:

```bash
python train.py --config configs/default_config.yaml
```

### 2. Export to ONNX

After training, export the best model:

```bash
python export_onnx.py \
  --checkpoint checkpoints/best_model.pth \
  --name my_character \
  --output ../voclo/models/my_character_sr44100_hop512_nfft2048.onnx
```

### 3. Test Model

Test the exported model:

```bash
python test_model.py \
  --model ../voclo/models/my_character_sr44100_hop512_nfft2048.onnx \
  --audio test.wav \
  --output output.wav
```

### 4. Use in voclo

Copy the model to `voclo/models/` (if not already there) and launch voclo.

## Model Requirements

### Inputs

The model should accept:
- **audio**: Float32 tensor [batch, time]
  - Audio signal in float32 format
  - Values in range [-1.0, 1.0]
  - Mono channel

### Outputs

The model should return:
- **output**: Float32 tensor [batch, time]
  - Converted audio signal
  - Same format as input

### Optional Inputs

The model may accept additional inputs:
- **speaker_id**: Int64 tensor [batch] - Speaker ID
- **pitch_shift**: Int64 or Float32 - Pitch shift
- **input_lengths**: Int64 tensor [batch] - Input sequence lengths

Voclo will automatically handle these inputs if they are present in the model.

## Troubleshooting

### Export Error

If export fails:
1. Make sure the model is fully trained
2. Check that all dependencies are installed
3. Try exporting without simplification: `--no-simplify`

### Model Doesn't Work in voclo

If the model doesn't load in voclo:
1. Check the filename format
2. Make sure parameters (sr, hop, nfft) are correct
3. Check voclo logs for errors
4. Make sure the model accepts correct inputs

### Poor Audio Quality

If audio quality is poor:
1. Check that the model is trained long enough
2. Make sure training data is of good quality
3. Try increasing the number of training epochs
4. Check that sample_rate matches

## Advanced Settings

### Custom Export Parameters

```bash
python export_onnx.py \
  --checkpoint checkpoints/best_model.pth \
  --sample-rate 48000 \
  --hop-length 512 \
  --n-fft 2048 \
  --opset-version 14 \
  --no-simplify
```

### Export with Multiple Speakers

If the model supports multiple speakers, make sure:
- `num_speakers` is correctly set in configuration
- The model accepts `speaker_id` as input
- Voclo can pass speaker_id (may require modification)

## Notes

- ONNX models are usually larger than PyTorch models
- Model simplification (`onnxsim`) can reduce size and speed up inference
- Always test the model before using in production
- Keep original PyTorch checkpoints for possible improvements
