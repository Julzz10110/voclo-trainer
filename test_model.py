"""Test exported ONNX model."""

import argparse
import onnx
import onnxruntime as ort
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

def test_onnx_model(model_path: str, test_audio_path: str = None, output_path: str = None):
    """Test an exported ONNX model.
    
    Args:
        model_path: Path to ONNX model
        test_audio_path: Path to test audio file (optional)
        output_path: Path to save output audio (optional)
    """
    print(f"Loading ONNX model: {model_path}")
    
    # Load and verify model
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model is valid")
    
    # Print model info
    print("\nModel inputs:")
    for input_info in onnx_model.graph.input:
        print(f"  - {input_info.name}: {[d.dim_value for d in input_info.type.tensor_type.shape.dim]}")
    
    print("\nModel outputs:")
    for output_info in onnx_model.graph.output:
        print(f"  - {output_info.name}: {[d.dim_value for d in output_info.type.tensor_type.shape.dim]}")
    
    # Create inference session
    session = ort.InferenceSession(model_path)
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"\nInput name: {input_name}")
    print(f"Output name: {output_name}")
    
    # Test with dummy input if no audio provided
    if test_audio_path is None:
        print("\nTesting with dummy input...")
        dummy_audio = np.random.randn(1, 4410).astype(np.float32)  # ~100ms at 44.1kHz
        
        outputs = session.run([output_name], {input_name: dummy_audio})
        output = outputs[0]
        
        print(f"✅ Inference successful!")
        print(f"   Input shape: {dummy_audio.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
    else:
        print(f"\nTesting with audio: {test_audio_path}")
        
        # Load audio
        audio, sr = librosa.load(test_audio_path, sr=44100, mono=True)
        audio = audio.astype(np.float32)
        
        # Prepare input (add batch dimension)
        audio_input = audio.reshape(1, -1)
        
        print(f"   Input audio: {len(audio)} samples ({len(audio)/sr:.2f}s)")
        
        # Run inference
        outputs = session.run([output_name], {input_name: audio_input})
        output_audio = outputs[0].squeeze()
        
        print(f"✅ Inference successful!")
        print(f"   Input shape: {audio_input.shape}")
        print(f"   Output shape: {output_audio.shape}")
        print(f"   Output range: [{output_audio.min():.4f}, {output_audio.max():.4f}]")
        
        # Save output if requested
        if output_path:
            sf.write(output_path, output_audio, sr)
            print(f"   Saved output to: {output_path}")
    
    print("\n✅ Model test completed!")


def main():
    parser = argparse.ArgumentParser(description="Test exported ONNX model")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to ONNX model file'
    )
    parser.add_argument(
        '--audio',
        type=str,
        default=None,
        help='Path to test audio file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output audio'
    )
    args = parser.parse_args()
    
    test_onnx_model(args.model, args.audio, args.output)


if __name__ == '__main__':
    main()

