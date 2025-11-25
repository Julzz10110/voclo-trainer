# Google Colab Training Guide

This is a detailed guide on using voclo-trainer in Google Colab.

## Colab Advantages

- ✅ Free GPU (T4, sometimes V100)
- ✅ No installation required on local computer
- ✅ Easy to share notebooks
- ✅ Automatic saving to Google Drive

## Quick Start

### 1. Open Notebook in Colab

1. Upload the `voclo_trainer_colab.ipynb` file to Google Drive
2. Open it via Google Colab (right-click → "Open with" → "Google Colab")
3. Or upload directly to Colab: File → Upload notebook

### 2. Enable GPU

1. Runtime → Change runtime type
2. Select "GPU" (T4 or better)
3. Save

### 3. Run Cells in Order

Follow the instructions in the notebook:
- Install dependencies
- Load data
- Configure
- Train
- Export
- Download

## Data Preparation

### Option 1: ZIP Archive

1. Create structure on your computer:
```
my_data/
  train/
    speaker1/
      audio1.wav
      audio2.wav
  val/
    speaker1/
      audio1.wav
```

2. Archive into `data.zip`
3. Upload via the "Load Data" cell in the notebook

### Option 2: Google Drive

1. Upload data to Google Drive
2. Mount Drive in the notebook
3. Specify the path to data

### Option 3: Direct File Upload

Use Colab's file manager to upload individual files.

## Loading Project Code

### Option 1: GitHub (Recommended)

If you have a GitHub repository:

```python
!git clone https://github.com/your-username/voclo-trainer.git
%cd voclo-trainer
```

### Option 2: Upload Files

1. Archive the `voclo-trainer` folder into ZIP
2. Upload to Colab via file manager
3. Extract:
```python
!unzip voclo-trainer.zip
%cd voclo-trainer
```

### Option 3: Create Files Inline

You can copy project file contents directly into notebook cells.

## Colab Configuration

### Optimal Parameters

For Colab, it's recommended:

```yaml
training:
  batch_size: 8-16  # Depends on available memory
  num_epochs: 50-100
  num_workers: 2  # Don't use more than 2 in Colab

data:
  segment_length: 8192  # Can reduce to 4096 to save memory
```

### Resource Usage Monitoring

Add to the notebook:

```python
# Check GPU memory
!nvidia-smi

# Check RAM usage
import psutil
print(f"RAM: {psutil.virtual_memory().percent}%")
```

## Training

### Basic Training

```python
!python train.py --config configs/colab_config.yaml
```

### Resume Training

If the session was interrupted:

```python
!python train.py \
  --config configs/colab_config.yaml \
  --resume checkpoints/model_epoch_50.pth
```

### Progress Monitoring

Add loss visualization:

```python
import matplotlib.pyplot as plt

# After training, if you saved logs
losses = [...]  # Your loss values
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
```

## Model Export

After training:

```python
!python export_onnx.py \
  --checkpoint checkpoints/best_model.pth \
  --name my_character \
  --output my_character_sr44100_hop512_nfft2048.onnx
```

## Saving Results

### Automatic Save to Drive

Add at the end of training:

```python
from google.colab import drive
import shutil

drive.mount('/content/drive')
backup_path = '/content/drive/MyDrive/voclo_backup/'

# Copy checkpoints
shutil.copytree('checkpoints', f'{backup_path}checkpoints', dirs_exist_ok=True)

# Copy ONNX
import glob
for onnx_file in glob.glob('*.onnx'):
    shutil.copy(onnx_file, backup_path)
```

### Download Files

```python
from google.colab import files

# Download ONNX model
files.download('my_character_sr44100_hop512_nfft2048.onnx')

# Download all checkpoints (ZIP)
import zipfile
with zipfile.ZipFile('checkpoints.zip', 'w') as zf:
    for file in Path('checkpoints').glob('*.pth'):
        zf.write(file)
files.download('checkpoints.zip')
```

## Troubleshooting

### Problem: Out of Memory

**Solution:**
1. Reduce `batch_size` to 4-8
2. Reduce `segment_length` to 4096
3. Use gradient accumulation:
```python
# In train.py add
accumulation_steps = 4
# And divide loss by accumulation_steps before backward
```

### Problem: Session Interrupted

**Solution:**
1. Save checkpoints frequently (`save_interval: 5`)
2. Use `--resume` to continue
3. Save to Google Drive automatically

### Problem: Slow Training

**Solution:**
1. Make sure GPU is used (not CPU)
2. Check: `torch.cuda.is_available()`
3. Increase `num_workers` (but not more than 2 in Colab)
4. Use Colab Pro for better GPU

### Problem: Data Not Loading

**Solution:**
1. Check directory structure
2. Make sure files have correct extensions (.wav, .mp3)
3. Check Google Drive access permissions

### Problem: Model Won't Export

**Solution:**
1. Make sure checkpoint is loaded correctly
2. Check ONNX version: `pip install onnx==1.14.0`
3. Try without simplification: `--no-simplify`

## Colab Optimization

### Using Colab Pro

Colab Pro provides:
- Longer sessions (up to 24 hours)
- Better GPUs (V100, A100)
- More memory

### Time Saving

1. **Preprocess data in advance**: Process audio on local computer
2. **Use ready checkpoints**: If you have a pre-trained model
3. **Reduce number of epochs**: Start with 50 epochs for testing

### Memory Saving

1. **Clear cache**:
```python
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
```

2. **Remove intermediate files**:
```python
!rm -rf __pycache__ *.pyc
```

## Useful Commands

### Check GPU
```python
!nvidia-smi
```

### Check Installed Packages
```python
!pip list | grep torch
!pip list | grep onnx
```

### Clear Everything
```python
!rm -rf checkpoints/*.pth
!rm -rf logs/*
```

## Automation

Create a script for the full cycle:

```python
# Full training cycle
import subprocess

# 1. Training
subprocess.run(['python', 'train.py', '--config', 'configs/colab_config.yaml'])

# 2. Export
subprocess.run(['python', 'export_onnx.py', 
                '--checkpoint', 'checkpoints/best_model.pth',
                '--name', 'my_model'])

# 3. Save
drive.mount('/content/drive')
shutil.copy('my_model.onnx', '/content/drive/MyDrive/')
```

## Additional Resources

- [Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [GPU Guide in Colab](https://colab.research.google.com/notebooks/gpu.ipynb)

## Support

If you encounter problems:
1. Check logs in the notebook
2. Make sure all dependencies are installed
3. Check library versions
4. Create an issue in the project repository
