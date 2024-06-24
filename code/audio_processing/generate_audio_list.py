"""
This script is used to generate a list of audio files in a directory and save it to a .lst file.

Usage:
    python generate_audio_list.py --audio_dir <audio_dir> --output <output_file>
    
    - audio_dir (str): The directory containing the audio files.
    - output (str): The path to save the list of audio files.
    
Example:
    python generate_audio_list.py --audio_dir /data/audio --output /data/metadata/wav_list.lst
"""

import os
import argparse


def generate_audio_list(audio_dir: str, output: str) -> None:
    """
    Generate a list of audio files in a directory and save it to a .lst file.

    Args:
    - audio_dir (str): The directory containing the audio files.
    - output (str): The path to save the list of audio files.
    """
    if not output.endswith('.lst'):
        output += '.lst'

    # Iterate through all directories and files in the audio directory
    audio_list = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                rel_path = os.path.relpath(os.path.join(root, file), audio_dir)
                audio_list.append(rel_path)

    # Save the list of audio files to a .lst file
    # If directory does not exist, create it
    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))

    with open(output, 'w') as f:
        for audio in audio_list:
            f.write(audio + '\n')

    print(f'Saved {len(audio_list)} audio files to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a list of audio files in a directory and save it to a .lst file.')
    parser.add_argument('--audio_dir', type=str,
                        help='The directory containing the audio files.')
    parser.add_argument('--output', type=str,
                        help='The path to save the list of audio files.')
    args = parser.parse_args()

    audio_dir = args.audio_dir
    output = args.output

    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f'The directory {audio_dir} does not exist.')

    generate_audio_list(audio_dir, output)
