"""Training script for voice conversion models."""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np

from models.rvc import RVCModel
from utils.dataset import VoiceConversionDataset
from utils.audio import extract_mel_spectrogram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> nn.Module:
    """Create model from configuration."""
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
    
    return model


def create_loss_fn(config: dict) -> nn.Module:
    """Create loss function from configuration."""
    loss_weights = config['training']['loss_weights']
    
    def combined_loss(pred, target, mel_pred, mel_target):
        """Combined loss function."""
        # Mel spectrogram loss
        mel_loss = nn.MSELoss()(mel_pred, mel_target)
        
        # Reconstruction loss (if applicable)
        recon_loss = nn.MSELoss()(pred, target)
        
        # Total loss
        total_loss = (
            loss_weights['mel'] * mel_loss +
            loss_weights['recon'] * recon_loss
        )
        
        return total_loss, {
            'mel_loss': mel_loss.item(),
            'recon_loss': recon_loss.item(),
            'total_loss': total_loss.item()
        }
    
    return combined_loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn,
    device: torch.device,
    config: dict
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_details = {}
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        audio = batch['audio'].to(device)
        mel = batch['mel'].to(device)
        f0 = batch['f0'].to(device)
        speaker_id = batch['speaker_id'].to(device)
        
        # Forward pass
        # For training, we use mel as input
        output = model(mel, speaker_id, f0)
        
        # Compute loss
        loss, details = loss_fn(
            output['mel'],
            mel,
            output['mel'],
            mel
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if 'gradient_clip' in config['training']:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['gradient_clip']
            )
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for key, value in details.items():
            if key not in loss_details:
                loss_details[key] = 0.0
            loss_details[key] += value
        
        # Update progress bar
        if batch_idx % config['logging'].get('console_log_interval', 100) == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mel_loss': f"{details['mel_loss']:.4f}"
            })
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    for key in loss_details:
        loss_details[key] /= len(dataloader)
    
    return {
        'loss': avg_loss,
        **loss_details
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn,
    device: torch.device,
    config: dict
) -> dict:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    loss_details = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            audio = batch['audio'].to(device)
            mel = batch['mel'].to(device)
            f0 = batch['f0'].to(device)
            speaker_id = batch['speaker_id'].to(device)
            
            # Forward pass
            output = model(mel, speaker_id, f0)
            
            # Compute loss
            loss, details = loss_fn(
                output['mel'],
                mel,
                output['mel'],
                mel
            )
            
            # Accumulate losses
            total_loss += loss.item()
            for key, value in details.items():
                if key not in loss_details:
                    loss_details[key] = 0.0
                loss_details[key] += value
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    for key in loss_details:
        loss_details[key] /= len(dataloader)
    
    return {
        'loss': avg_loss,
        **loss_details
    }


def main():
    parser = argparse.ArgumentParser(description="Train voice conversion model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    logger.info(f"Created model: {model.__class__.__name__}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    train_dataset = VoiceConversionDataset(
        data_dir=config['data']['train_dir'],
        sample_rate=config['model']['sample_rate'],
        segment_length=config['data']['segment_length'],
        hop_length=config['model']['hop_length'],
        n_fft=config['model']['n_fft'],
        n_mels=config['model']['n_mels'],
        f0_min=config['model'].get('f0_min', 50.0),
        f0_max=config['model'].get('f0_max', 1100.0),
        augmentation=config['data'].get('augmentation'),
        cache=False
    )
    
    val_dataset = VoiceConversionDataset(
        data_dir=config['data']['val_dir'],
        sample_rate=config['model']['sample_rate'],
        segment_length=config['data']['segment_length'],
        hop_length=config['model']['hop_length'],
        n_fft=config['model']['n_fft'],
        n_mels=config['model']['n_mels'],
        f0_min=config['model'].get('f0_min', 50.0),
        f0_max=config['model'].get('f0_max', 1100.0),
        augmentation=None,  # No augmentation for validation
        cache=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True)
    )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0)
    )
    
    # Create loss function
    loss_fn = create_loss_fn(config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    save_interval = config['training']['save_interval']
    val_interval = config['training'].get('validation_interval', 5)
    
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, config
        )
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        
        # Validate
        if (epoch + 1) % val_interval == 0:
            val_metrics = validate(
                model, val_loader, loss_fn, device, config
            )
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_path = checkpoint_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config
                }, best_path)
                logger.info(f"Saved best model to {best_path}")
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = checkpoint_dir / f'model_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()

