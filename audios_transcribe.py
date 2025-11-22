#!/usr/bin/env python

# pip install torch transformers torchaudio accelerate librosa
# require ffmpeg binary in PATH

import os
import argparse
import torch
from transformers import pipeline
from tqdm import tqdm
import logging

# Suppress HuggingFace warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# The specialized model for Anime/Game speech
MODEL_ID = "litagin/anime-whisper"

def main():
    parser = argparse.ArgumentParser(description="Batch transcribe audio files to .txt using Anime Whisper.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing .wav files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .txt files if they exist")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    args = parser.parse_args()

    # 1. Validate Input Directory
    if not os.path.isdir(args.input_dir):
        print(f"[!] Error: Directory '{args.input_dir}' not found.")
        return

    # 2. Scan files and determine work list
    all_wavs = [
        os.path.join(args.input_dir, f) 
        for f in os.listdir(args.input_dir) 
        if f.lower().endswith(".wav")
    ]
    all_wavs.sort()

    files_to_process = []
    skipped_count = 0

    for wav_path in all_wavs:
        # Generate expected text filename:  audio.wav -> audio.txt
        txt_path = os.path.splitext(wav_path)[0] + ".txt"
        
        if os.path.exists(txt_path) and not args.force:
            skipped_count += 1
        else:
            files_to_process.append(wav_path)

    print(f"[+] Found {len(all_wavs)} .wav files.")
    print(f"[+] Skipped {skipped_count} files (transcripts already exist).")
    print(f"[+] Queued {len(files_to_process)} files for transcription.")

    if not files_to_process:
        print("[✓] Nothing to do.")
        return

    # 3. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Loading Model: {MODEL_ID} on {device}...")

    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_ID,
            chunk_length_s=30,
            device=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        return

    # 4. Batch Process
    print("[+] Starting transcription...")
    
    # Process in chunks based on batch_size
    for i in tqdm(range(0, len(files_to_process), args.batch_size), desc="Processing", unit="batch"):
        batch_files = files_to_process[i : i + args.batch_size]
        
        try:
            # Run inference
            outputs = pipe(
                batch_files, 
                generate_kwargs={"language": "japanese", "task": "transcribe"}, 
                batch_size=args.batch_size
            )
            
            # Save outputs immediately
            for wav_path, output in zip(batch_files, outputs):
                text = output["text"].strip()
                txt_path = os.path.splitext(wav_path)[0] + ".txt"
                
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)

        except Exception as e:
            print(f"\n[!] Error processing batch starting at {batch_files[0]}: {e}")
            continue

    print("\n[✓] Transcription complete.")

if __name__ == "__main__":
    main()
