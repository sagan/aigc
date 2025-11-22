#!/usr/bin/env python

# pip install speechbrain "torchaudio<2.9.0" umap-learn hdbscan matplotlib joblib soundfile tqdm

import os
import sys
import shutil
import numpy as np
import joblib
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
from speechbrain.inference import EncoderClassifier
import umap
import hdbscan

# ========================
# Configuration parameters
# ========================
EMBEDDINGS_CACHE_NAME = "embeddings_ecapa.joblib"
MIN_DURATION_SEC = 1              # ECAPA is robust, can handle slightly shorter clips than Resemblyzer
MIN_CLUSTER_SIZE = 15               # Minimum files to form a valid speaker cluster
CONFIDENCE_THRESHOLD = 0.5         # Probability threshold to keep files in a cluster
UMAP_COMPONENTS = 15                # Reduce 192-dim embedding to 15-dim for clustering
UMAP_NEIGHBORS = 30                 # Local neighborhood size for UMAP
# ========================

def load_model():
    """Load SpeechBrain ECAPA-TDNN model."""
    print("[+] Loading SpeechBrain ECAPA-TDNN model...")
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps" 
    
    print(f"[+] Inference device: {device}")
    
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        run_opts={"device": device}
    )
    return classifier, device

def preprocess_audio(file_path):
    """Load, resample to 16k, and mix to mono."""
    try:
        signal, fs = torchaudio.load(file_path)
        
        # 1. Convert to Mono
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
            
        # 2. Resample to 16000 Hz (Required for ECAPA-TDNN)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)
            
        return signal
    except Exception as e:
        print(f"[!] Error loading {file_path}: {e}")
        return None

def compute_or_load_embeddings(input_dir, cache_path):
    """Compute embeddings using SpeechBrain or load from cache."""
    if os.path.exists(cache_path):
        print(f"[+] Cached embeddings found. Loading from {cache_path} ...")
        data = joblib.load(cache_path)
        return data["wav_files"], data["embeddings"]

    classifier, device = load_model()
    print("[+] No cache found. Computing embeddings from WAV files...")

    # Gather WAV files
    wav_files = sorted(
        [os.path.join(input_dir, f)
         for f in os.listdir(input_dir)
         if f.lower().endswith(".wav")]
    )

    print(f"[+] Found {len(wav_files)} WAV files. Filtering and Extracting...")
    
    valid_files = []
    embeddings = []
    short_files = []

    # Process Loop
    for f in tqdm(wav_files, ncols=80):
        # Quick duration check first using soundfile (faster than loading tensor)
        try:
            info = sf.info(f)
            if info.duration < MIN_DURATION_SEC:
                short_files.append(f)
                print(f"Skip short file {f} (duration: {info.duration})")
                continue
        except:
            continue

        # Load and Preprocess
        wav_tensor = preprocess_audio(f)
        if wav_tensor is None:
            print(f"Error loading {f}: parsed wav_tensor is None")
            continue

        # Extract Embedding
        # Move tensor to GPU for inference
        wav_tensor = wav_tensor.to(device)
        
        with torch.no_grad():
            # encode_batch expects a batch, but we pass 1 file. 
            # Output shape is (1, 1, 192) -> squeeze to (192)
            emb = classifier.encode_batch(wav_tensor).squeeze().cpu().numpy()
            
        embeddings.append(emb)
        valid_files.append(f)

    print(f"[+] {len(valid_files)} embeddings extracted. {len(short_files)} skipped (too short).")

    # Save Cache
    embeddings = np.vstack(embeddings)
    data = {"wav_files": valid_files, "embeddings": embeddings}
    joblib.dump(data, cache_path)
    print(f"[✓] Saved embeddings cache → {cache_path}")
    
    return valid_files, embeddings

def cluster_embeddings(wav_files, embeddings, output_dir):
    """Cluster using UMAP reduction + HDBSCAN."""
    
    # 1. Dimensionality Reduction using UMAP (Better than PCA for speakers)
    print(f"[+] Reducing dimensions with UMAP (Neighbors={UMAP_NEIGHBORS})...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        n_components=UMAP_COMPONENTS,
        min_dist=0.0,
        metric='cosine', # Cosine distance is better for embeddings
        random_state=42
    )
    X_embedded = reducer.fit_transform(embeddings)

    # 2. Clustering with HDBSCAN
    print("[+] Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=5,
        metric='euclidean', # Euclidean on UMAP space works well
        cluster_selection_method='eom',
        prediction_data=True
    )
    labels = clusterer.fit_predict(X_embedded)
    
    # Calculate probabilities (confidence)
    probs = clusterer.probabilities_

    unique_labels = sorted(set(labels) - {-1}, key=lambda l: np.sum(labels == l), reverse=True)
    noise_count = np.sum(labels == -1)
    print(f"[+] Found {len(unique_labels)} clusters. (Noise/Unknown files: {noise_count})")

    # 3. Sort and Move Files
    # Sort clusters by size
    cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
    sorted_clusters = sorted(unique_labels, key=lambda l: cluster_sizes[l], reverse=True)

    print("[+] Organizing files into folders...")
    for i, label in enumerate(sorted_clusters):
        # Create folder: SPEAKER_00_Count1400
        count = cluster_sizes[label]
        speaker_dir = os.path.join(output_dir, f"SPEAKER_{i:02d}_count{count}")
        os.makedirs(speaker_dir, exist_ok=True)
        
        member_indices = np.where(labels == label)[0]
        
        # Filter by confidence
        high_conf_indices = member_indices[probs[member_indices] >= CONFIDENCE_THRESHOLD]
        
        for idx in high_conf_indices:
            src = wav_files[idx]
            dst = os.path.join(speaker_dir, os.path.basename(src))
            shutil.copy2(src, dst)

    # Handle Noise / Low Confidence
    noise_dir = os.path.join(output_dir, "UNSORTED_NOISE")
    os.makedirs(noise_dir, exist_ok=True)
    
    noise_indices = np.where((labels == -1) | (probs < CONFIDENCE_THRESHOLD))[0]
    # Optional: Uncomment to copy noise files (can take up space)
    # for idx in noise_indices:
    #     src = wav_files[idx]
    #     dst = os.path.join(noise_dir, os.path.basename(src))
    #     shutil.copy2(src, dst)

    print(f"[✓] Done. Check {output_dir}")
    print(f"    - Identified Clusters: {len(unique_labels)}")
    print(f"    - Unsorted/Noise: {len(noise_indices)}")

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, EMBEDDINGS_CACHE_NAME)

    wav_files, embeddings = compute_or_load_embeddings(input_dir, cache_path)
    cluster_embeddings(wav_files, embeddings, output_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_dir, output_dir)
