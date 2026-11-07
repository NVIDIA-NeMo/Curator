#!/usr/bin/env python3
"""
Resample folder script using torchaudio
Resamples audio files from source folder to target sample rate
Based on DCASE task baseline requirements (typically 44kHz -> 16kHz)
"""

import argparse
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F
from tqdm import tqdm

torchaudio.set_audio_backend("soundfile")


def resample_file(input_file, output_file, target_sr=16000, overwrite=False):
    """
    Resample a single audio file using torchaudio

    Args:
        input_file (str): Path to input audio file
        output_file (str): Path to output audio file
        target_sr (int): Target sample rate (default: 16000)
        overwrite (bool): Whether to overwrite existing files

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Skip if output exists and overwrite is False
        if os.path.exists(output_file) and not overwrite:
            return True

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Load audio file
        waveform, orig_sr = torchaudio.load(input_file)

        # Skip resampling if already at target sample rate
        if orig_sr == target_sr:
            torchaudio.save(output_file, waveform, target_sr)
            return True

        # Resample using torchaudio functional
        resampled_waveform = F.resample(
            waveform,
            orig_freq=orig_sr,
            new_freq=target_sr,
            resampling_method="sinc_interpolation",
        )

        # Save resampled audio
        torchaudio.save(
            output_file,
            resampled_waveform,
            target_sr,
            bits_per_sample=16,
            encoding="PCM_S",
        )

        return True

    except Exception as e:
        warnings.warn(f"Error processing {input_file}: {str(e)}")
        return False


def resample_folder(
    input_folder,
    output_folder,
    target_sr=16000,
    audio_extensions=None,
    overwrite=False,
    num_workers=4,
    preserve_structure=True,
):
    """
    Resample all audio files in a folder using torchaudio

    Args:
        input_folder (str): Path to input folder
        output_folder (str): Path to output folder
        target_sr (int): Target sample rate (default: 16000)
        audio_extensions (list): List of audio file extensions to process
        overwrite (bool): Whether to overwrite existing files
        num_workers (int): Number of parallel workers
        preserve_structure (bool): Whether to preserve folder structure

    Returns:
        dict: Summary of processing results
    """
    if audio_extensions is None:
        audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]

    # Convert to Path objects
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_path.rglob(f"*{ext}"))
        audio_files.extend(input_path.rglob(f"*{ext.upper()}"))

    if not audio_files:
        print(f"No audio files found in {input_folder}")
        return {"processed": 0, "errors": 0, "skipped": 0}

    print(f"Found {len(audio_files)} audio files to process")

    # Prepare file pairs for processing
    file_pairs = []
    for input_file in audio_files:
        if preserve_structure:
            # Maintain relative path structure
            rel_path = input_file.relative_to(input_path)
            output_file = output_path / rel_path
        else:
            # Flat structure in output folder
            output_file = output_path / input_file.name

        # Ensure output has .wav extension
        output_file = output_file.with_suffix(".wav")
        file_pairs.append((str(input_file), str(output_file)))

    # Process files
    processed = 0
    errors = 0
    skipped = 0

    if num_workers > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    resample_file, input_f, output_f, target_sr, overwrite
                ): (input_f, output_f)
                for input_f, output_f in file_pairs
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(file_pairs), desc="Resampling") as pbar:
                for future in as_completed(future_to_file):
                    input_f, output_f = future_to_file[future]
                    try:
                        success = future.result()
                        if success:
                            if os.path.exists(output_f) and not overwrite:
                                skipped += 1
                            else:
                                processed += 1
                        else:
                            errors += 1
                    except Exception as e:
                        warnings.warn(f"Error processing {input_f}: {str(e)}")
                        errors += 1
                    finally:
                        pbar.update(1)
    else:
        # Sequential processing
        with tqdm(file_pairs, desc="Resampling") as pbar:
            for input_file, output_file in pbar:
                if os.path.exists(output_file) and not overwrite:
                    skipped += 1
                    continue

                success = resample_file(input_file, output_file, target_sr, overwrite)
                if success:
                    processed += 1
                else:
                    errors += 1

    # Print summary
    print(f"\nResampling complete!")
    print(f"Processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")

    return {"processed": processed, "errors": errors, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(
        description="Resample audio files in a folder using torchaudio"
    )
    parser.add_argument(
        "input_folder", type=str, help="Path to input folder containing audio files"
    )
    parser.add_argument(
        "output_folder", type=str, help="Path to output folder for resampled files"
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=16000,
        help="Target sample rate (default: 16000)",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".wav", ".mp3", ".flac", ".m4a", ".ogg"],
        help="Audio file extensions to process",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--flat_structure",
        action="store_true",
        help="Don't preserve folder structure in output",
    )

    args = parser.parse_args()

    # Validate input folder
    if not os.path.exists(args.input_folder):
        raise ValueError(f"Input folder does not exist: {args.input_folder}")

    # Resample folder
    results = resample_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        target_sr=args.target_sr,
        audio_extensions=args.extensions,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        preserve_structure=not args.flat_structure,
    )

    return results


if __name__ == "__main__":
    main()
