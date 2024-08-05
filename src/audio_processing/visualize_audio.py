"""
This file is used to visualize audio data.

Usage:
    python visualize_audio.py --audio_path <audio_path> --output_dir <output_dir>
    
    - audio_path (str): The path to the audio file.
    - output_dir (str): The directory to save the visualization.
    - start_time (float): The start time of the audio segment to visualize.
    - end_time (float): The end time of the audio segment to visualize.
    
Example:
    python visualize_audio.py --audio_path /data/audio/sample.wav --output_dir /data/visualization --start_time 0 --end_time 5
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import argparse

def visualize_audio(audio_path: str, output_dir: str, start_time: float = 0, end_time: float = None) -> None:
    """
    Visualize audio data.

    Args:
    - audio_path (str): The path to the audio file.
    - output_dir (str): The directory to save the visualization.
    - start_time (float): The start time of the audio segment to visualize.
    - end_time (float): The end time of the audio segment to visualize.
    
    Returns:
    - None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the audio file
    print("Loading audio file...")
    y, sr = librosa.load(audio_path, sr=None)

    # Create the time axis
    t = np.arange(0, len(y)) / sr
    start_idx = int(start_time * sr)
    end_idx = None if end_time is None else int(end_time * sr)
    y = y[start_idx:end_idx]
    t = t[start_idx:end_idx]

    # Plot the audio waveform
    plt.figure(figsize=(10, 4))
    plt.plot(t, y, color='grey', linewidth=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, f'{os.path.basename(audio_path).replace(".wav", "_waveform")}_{int(start_time)}_to_{int(end_time)}s.png')
    plt.savefig(output_path)
    print(f'Audio waveform saved to {output_path}.')

    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize audio data.')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='The path to the audio file.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='The directory to save the visualization.')
    parser.add_argument('--start_time', type=float, default=0,
                        help='The start time of the audio segment to visualize.')
    parser.add_argument('--end_time', type=float, default=None,
                        help='The end time of the audio segment to visualize.')
    
    args = parser.parse_args()
    
    visualize_audio(args.audio_path, args.output_dir, args.start_time, args.end_time)
    