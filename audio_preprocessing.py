"""
Enhanced Audio Preprocessing Module for Fake Audio Detection
Supports multiple audio formats with automatic conversion to WAV
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
import subprocess
import tempfile
import shutil
from pathlib import Path

# Supported audio formats
SUPPORTED_FORMATS = {
    'native': ['.wav', '.flac', '.ogg', '.m4a'],  # Librosa can handle these directly
    'requires_conversion': ['.mp3', '.mp4', '.aac', '.wma', '.aiff', '.webm', '.mov', '.mkv']
}

def check_ffmpeg_installed():
    """
    Check if FFmpeg is installed and accessible from the system PATH.
    
    Returns:
        bool: True if FFmpeg is available, False otherwise
    """
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def convert_to_wav(input_path, sr=16000):
    """
    Convert any audio format to WAV using FFmpeg.
    
    Args:
        input_path (str): Path to the input audio file
        sr (int): Target sample rate (default: 16000 Hz)
    
    Returns:
        str: Path to the temporary WAV file, or None if conversion failed
    """
    try:
        # Create a temporary file for the WAV output
        temp_wav_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        
        # FFmpeg command to convert audio to WAV
        command = [
            'ffmpeg',
            '-i', str(input_path),      # Input file
            '-vn',                      # Disable video
            '-acodec', 'pcm_s16le',     # Audio codec (16-bit PCM)
            '-ar', str(sr),             # Audio sample rate
            '-ac', '1',                 # Audio channels (mono)
            '-y',                       # Overwrite output file
            '-loglevel', 'error',       # Suppress verbose output
            temp_wav_file
        ]
        
        # Execute FFmpeg command
        result = subprocess.run(command, 
                              check=True, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              timeout=60)
        
        # Verify the file was created
        if os.path.exists(temp_wav_file) and os.path.getsize(temp_wav_file) > 0:
            return temp_wav_file
        else:
            print(f"Error: FFmpeg conversion resulted in an empty or missing file")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion error: {e.stderr.decode() if e.stderr else str(e)}")
        return None
    except subprocess.TimeoutExpired:
        print(f"FFmpeg conversion timed out. The audio file may be too large or corrupted.")
        return None
    except Exception as e:
        print(f"Unexpected error during FFmpeg conversion: {str(e)}")
        return None

def extract_features(audio_path, sr=16000, n_mfcc=40, n_mels=128, hop_length=512, n_fft=2048):
    """
    Extracts audio features for fake audio detection.
    Automatically handles multiple audio formats by converting to WAV if needed.
    
    Args:
        audio_path (str): Path to the audio file (supports WAV, MP3, MP4, AAC, etc.)
        sr (int): Sample rate for audio loading (default: 16000 Hz)
        n_mfcc (int): Number of MFCC coefficients (default: 40)
        n_mels (int): Number of mel bands (default: 128)
        hop_length (int): Number of samples between successive frames (default: 512)
        n_fft (int): FFT window size (default: 2048)
    
    Returns:
        dict: Dictionary containing extracted features, or None if extraction failed
    """
    
    # Validate input file
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return None
    
    file_ext = os.path.splitext(audio_path)[1].lower()
    temp_wav_file = None
    audio_path_to_process = audio_path
    
    print(f"\n{'='*70}")
    print(f"Processing audio file: {os.path.basename(audio_path)}")
    print(f"File format: {file_ext}")
    print(f"{'='*70}")
    
    # Check if format conversion is needed
    if file_ext not in SUPPORTED_FORMATS['native']:
        print(f"\n[1/4] Format conversion required ({file_ext} → WAV)")
        
        # Check if FFmpeg is available
        if not check_ffmpeg_installed():
            print("Error: FFmpeg is not installed or not in system PATH.")
            print("Please install FFmpeg to process non-WAV audio files.")
            print("Installation instructions:")
            print("  Windows: choco install ffmpeg  (or download from ffmpeg.org)")
            print("  macOS:   brew install ffmpeg")
            print("  Linux:   sudo apt install ffmpeg")
            return None
        
        # Convert to WAV
        print(f"Converting {file_ext} to temporary WAV file...")
        temp_wav_file = convert_to_wav(audio_path, sr)
        
        if temp_wav_file is None:
            print(f"Error: Failed to convert {file_ext} file to WAV")
            return None
        
        audio_path_to_process = temp_wav_file
        print(f"✓ Conversion successful: {temp_wav_file}")
    else:
        print(f"\n[1/4] Format supported natively: {file_ext}")
    
    try:
        # Load the audio file
        print(f"\n[2/4] Loading audio file...")
        y, sr_loaded = librosa.load(audio_path_to_process, sr=sr)
        print(f"✓ Audio loaded successfully")
        print(f"  Duration: {len(y) / sr_loaded:.2f} seconds")
        print(f"  Sample rate: {sr_loaded} Hz")
        
        # Extract features
        print(f"\n[3/4] Extracting audio features...")
        
        # MFCC (Mel-Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr_loaded, n_mfcc=n_mfcc, 
                                     hop_length=hop_length, n_fft=n_fft)
        
        # Mel-Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr_loaded, 
                                                         n_mels=n_mels, 
                                                         hop_length=hop_length, 
                                                         n_fft=n_fft)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # STFT (Short-Time Fourier Transform)
        stft = librosa.stft(y, hop_length=hop_length, n_fft=n_fft)
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        # Pitch estimation
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr_loaded, S=mel_spectrogram)
        pitch_estimate = np.array([pitches[magnitudes[:, t].argmax(), t] 
                                  for t in range(magnitudes.shape[1])])
        
        # RMS Energy
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        
        print(f"✓ Features extracted successfully")
        print(f"  MFCC shape: {mfccs.shape}")
        print(f"  Mel-Spectrogram shape: {mel_spectrogram_db.shape}")
        print(f"  STFT shape: {stft_db.shape}")
        
        # Normalize features
        print(f"\n[4/4] Normalizing features...")
        mfccs_normalized = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / \
                          (np.std(mfccs, axis=1, keepdims=True) + 1e-6)
        print(f"✓ Normalization complete")
        
        print(f"\n{'='*70}")
        print(f"✓ Feature extraction completed successfully!")
        print(f"{'='*70}\n")
        
        # Return all extracted features
        return {
            'mfccs': mfccs,
            'mfccs_normalized': mfccs_normalized,
            'mel_spectrogram_db': mel_spectrogram_db,
            'stft_db': stft_db,
            'pitch_estimate': pitch_estimate,
            'rms': rms,
            'raw_audio': y,
            'sample_rate': sr_loaded,
            'duration': len(y) / sr_loaded
        }
        
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        return None
    
    finally:
        # Clean up temporary file if it was created
        if temp_wav_file and os.path.exists(temp_wav_file):
            try:
                os.remove(temp_wav_file)
                print(f"Cleaned up temporary file")
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {str(e)}")

def visualize_features(features, output_dir="features_viz"):
    """
    Visualizes the extracted features and saves them as images.
    
    Args:
        features (dict): Dictionary of extracted features from extract_features()
        output_dir (str): Directory to save visualization images
    """
    if not features:
        print("Error: No features to visualize")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. MFCCs
        print(f"Generating MFCC visualization...")
        plt.figure(figsize=(12, 5))
        librosa.display.specshow(features['mfccs_normalized'], x_axis='time', sr=features['sample_rate'])
        plt.colorbar(label='Normalized MFCC')
        plt.title('Normalized MFCCs')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mfccs.png'), dpi=150)
        plt.close()
        print(f"  ✓ Saved: mfccs.png")
        
        # 2. Mel-Spectrogram
        print(f"Generating Mel-Spectrogram visualization...")
        plt.figure(figsize=(12, 5))
        librosa.display.specshow(features['mel_spectrogram_db'], x_axis='time', 
                                y_axis='mel', sr=features['sample_rate'])
        plt.colorbar(format='%+2.0f dB', label='Power (dB)')
        plt.title('Mel-Spectrogram (dB)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mel_spectrogram.png'), dpi=150)
        plt.close()
        print(f"  ✓ Saved: mel_spectrogram.png")
        
        # 3. STFT Spectrogram
        print(f"Generating STFT Spectrogram visualization...")
        plt.figure(figsize=(12, 5))
        librosa.display.specshow(features['stft_db'], x_axis='time', 
                                y_axis='log', sr=features['sample_rate'])
        plt.colorbar(format='%+2.0f dB', label='Magnitude (dB)')
        plt.title('STFT Spectrogram (dB)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stft_spectrogram.png'), dpi=150)
        plt.close()
        print(f"  ✓ Saved: stft_spectrogram.png")
        
        # 4. RMS Energy
        print(f"Generating RMS Energy visualization...")
        plt.figure(figsize=(12, 4))
        frames = range(len(features['rms']))
        t = librosa.frames_to_time(frames, sr=features['sample_rate'])
        plt.plot(t, features['rms'], label='RMS Energy')
        plt.xlabel('Time (s)')
        plt.ylabel('RMS Energy')
        plt.title('RMS Energy Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rms_energy.png'), dpi=150)
        plt.close()
        print(f"  ✓ Saved: rms_energy.png")
        
        print(f"\n✓ All visualizations saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")

# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("AUDIO PREPROCESSING MODULE - TEST")
    print("="*70)
    
    # Check FFmpeg availability
    print("\nChecking FFmpeg installation...")
    if check_ffmpeg_installed():
        print("✓ FFmpeg is installed and accessible")
    else:
        print("⚠ FFmpeg is not installed. Multi-format support will be limited.")
    
    print("\nSupported audio formats:")
    print(f"  Native support: {', '.join(SUPPORTED_FORMATS['native'])}")
    print(f"  Requires FFmpeg: {', '.join(SUPPORTED_FORMATS['requires_conversion'])}")
