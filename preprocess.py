"""Data preprocessing utilities for voice conversion training."""

import argparse
import os
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_audio_file(
    input_path: str,
    output_path: str,
    target_sr: int = 44100,
    normalize: bool = True,
    trim_silence: bool = True
) -> bool:
    """Preprocess a single audio file.
    
    Args:
        input_path: Input audio file
        output_path: Output audio file
        target_sr: Target sample rate
        normalize: Normalize audio
        trim_silence: Trim leading/trailing silence
        
    Returns:
        True if successful
    """
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
        
        # Trim silence
        if trim_silence:
            audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Normalize
        if normalize:
            max_val = abs(audio).max()
            if max_val > 0:
                audio = audio / max_val
        
        # Save
        sf.write(output_path, audio, target_sr)
        return True
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        return False


def preprocess_directory(
    input_dir: str,
    output_dir: str,
    target_sr: int = 44100,
    normalize: bool = True,
    trim_silence: bool = True
):
    """Preprocess all audio files in a directory.
    
    Maintains directory structure.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(input_path.rglob(f'*{ext}'))
        audio_files.extend(input_path.rglob(f'*{ext.upper()}'))
    
    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Process each file
    success_count = 0
    for audio_file in tqdm(audio_files, desc="Preprocessing"):
        # Calculate relative path
        rel_path = audio_file.relative_to(input_path)
        output_file = output_path / rel_path.with_suffix('.wav')
        
        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Process
        if preprocess_audio_file(
            str(audio_file),
            str(output_file),
            target_sr,
            normalize,
            trim_silence
        ):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(audio_files)} files")


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio data for training")
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory with audio files'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for preprocessed files'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=44100,
        help='Target sample rate'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Disable audio normalization'
    )
    parser.add_argument(
        '--no-trim',
        action='store_true',
        help='Disable silence trimming'
    )
    args = parser.parse_args()
    
    preprocess_directory(
        args.input,
        args.output,
        target_sr=args.sample_rate,
        normalize=not args.no_normalize,
        trim_silence=not args.no_trim
    )
    
    logger.info("Preprocessing completed!")


if __name__ == '__main__':
    main()

