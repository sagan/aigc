#/usr/bin/env python

import os
import re
import subprocess
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from typing import List, Dict, TypedDict, Optional, Tuple, Iterable, Set, Pattern, Final

from tqdm import tqdm
import pandas as pd


# ========== ARGUMENT PARSER ==========

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ViSQOL evaluation on generated audio files.")

    parser.add_argument("--gen-audio-dir", type=str, default="/data/kotomi-evaluation/output")
    parser.add_argument("--ref-audio-dir", type=str, default="/data/kotomi-evaluation/input")
    parser.add_argument("--candidates-dir", type=str, default="/data/kotomi-evaluation/candidates")
    parser.add_argument("--visqol-model-path", type=str, default="/data/visqol-model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite")
    parser.add_argument("--cache-file", type=str, default="visqol_scores.json")
    parser.add_argument("--ref-audio-suffix", type=str, default="", help="reference audio filename suffix, e.g. '_32k'")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--max-threads", type=int, default=64)

    return parser.parse_args()


# Read script directory
script_directory: str = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)


# ========== TYPE DEFINITIONS ==========

class ModelScore(TypedDict):
    Model: str
    Reference: str
    Score: float

class TaskDict(TypedDict):
    model_id: str
    gpt_ep: int
    sovits_ep: int
    ref_id: str
    ref_path: str
    deg_path: str
    score: Optional[float]

class ResultEntry(TypedDict):
    Model: str
    GPT_Ep: int
    SoVITS_Ep: int
    Reference: str
    Score: float


FILENAME_PATTERN: Final[Pattern[str]] = re.compile(
    r"(.+?_gpt(\d+)_sovits(\d+))_(.+)\.wav"
)


# ========== CACHE FUNCTIONS ==========

def load_cache(filepath: str) -> List[ModelScore]:
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data: List[ModelScore] = json.load(f)
                return data
        except Exception as e:
            print(f"Warning: Failed to load cache file: {e}")
    return []


def save_cache(filepath: str, data: List[ModelScore]) -> None:
    temp_filepath: str = filepath + ".tmp"
    try:
        with open(temp_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_filepath, filepath)
    except Exception as e:
        print(f"Warning: Failed to save cache file: {e}")
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass


# ========== PROCESSING FUNCTION ==========

def process_single_pair(task: TaskDict, visqol_model_path: str) -> TaskDict:
    cmd: List[str] = [
        "visqol", "--use_speech_mode",
        "--reference_file", task['ref_path'],
        "--degraded_file", task['deg_path'],
        "--similarity_to_quality_model", visqol_model_path,
        "--use_lattice_model=false"
    ]

    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        match: Optional[re.Match[str]] = re.search(
            r"MOS-LQO:\s+([0-9.]+)", result.stdout
        )
        task['score'] = float(match.group(1)) if match else None
    except Exception:
        task['score'] = None

    return task


# ========== COPY CANDIDATES ==========

def copy_candidates(target_models: Iterable[str], all_files: List[str],
                    gen_audio_dir: str, candidates_dir: str) -> None:

    if not target_models:
        print("No models selected for copying.")
        return

    print("\n" + "=" * 50)
    print("📂 COPYING CANDIDATES")
    print("=" * 50)

    os.makedirs(candidates_dir, exist_ok=True)

    copied = skipped = missing = 0

    for model_id in target_models:
        prefix: str = model_id + "_"
        matches: List[str] = [
            f for f in all_files
            if f.startswith(prefix) and f.endswith(".wav") and ".tmp" not in f
        ]

        if not matches:
            missing += 1
            continue

        for f in matches:
            src: str = os.path.join(gen_audio_dir, f)
            dst: str = os.path.join(candidates_dir, f)

            if os.path.exists(dst):
                skipped += 1
            else:
                try:
                    os.link(src, dst)
                    copied += 1
                except OSError as e:
                    print(f"  [Error] Failed to link {f}: {e}")

    print(f"Linked: {copied} | Skipped: {skipped} | Missing: {missing}")
    print(f"Location: {candidates_dir}")


# ========== ANALYSIS FUNCTIONS ==========

def analyze_distribution(df: pd.DataFrame, title: str) -> pd.DataFrame:
    print(f"\n--- {title} ---")
    if df.empty:
        print("No data.")
        return df

    count = len(df)
    mean_val: float = float(df["Average"].mean())
    std_val: float = float(df["Average"].std())

    print(f"Count: {count} | Mean: {mean_val:.4f} | SD: {std_val:.4f}")
    print("-" * 50)

    sorted_df = df.sort_values("Average", ascending=False) # type: ignore

    print("📈 Top 3:")
    for i, (idx, row) in enumerate(sorted_df.head(3).iterrows(), 1): # type: ignore
        print(f"  {i}. {idx:<30} | {row['Average']:.4f}")

    if count > 3:
        print("📉 Bottom 3:")
        tail = sorted_df.tail(3)
        start_rank = count - len(tail) + 1

        for i, (idx, row) in enumerate(tail.iterrows(), start_rank): # type: ignore
            print(f"  {i}. {idx:<30} | {row['Average']:.4f}")

    return sorted_df.head(3)


# ========== MAIN FUNCTION ==========

def main() -> None:
    args = parse_args()

    GEN_AUDIO_DIR:str = args.gen_audio_dir
    REF_AUDIO_DIR:str = args.ref_audio_dir
    CANDIDATES_DIR:str = args.candidates_dir
    VISQOL_MODEL_PATH:str = args.visqol_model_path
    CACHE_FILE:str = args.cache_file
    SAVE_INTERVAL:int = args.save_interval
    MAX_THREADS:int = args.max_threads
    REF_AUDIO_SUFFIX:str = args.ref_audio_suffix

    cache_data = load_cache(CACHE_FILE)
    cache_map: Dict[Tuple[str, str], float] = {
        (item['Model'], item['Reference']): item['Score'] for item in cache_data
    }

    if not os.path.exists(GEN_AUDIO_DIR):
        print(f"Error: {GEN_AUDIO_DIR} not found.")
        return

    gen_files: List[str] = [
        f for f in os.listdir(GEN_AUDIO_DIR)
        if f.endswith(".wav") and ".tmp" not in f
    ]
    print(f"Scanning {len(gen_files)} generated files...")

    tasks_to_run: List[TaskDict] = []
    final_results: List[ResultEntry] = []

    for f in gen_files:
        match: Optional[re.Match[str]] = FILENAME_PATTERN.match(f)
        if not match:
            continue

        model_id = match.group(1)
        gpt_ep = int(match.group(2))
        sovits_ep = int(match.group(3))
        ref_id = match.group(4)

        if (model_id, ref_id) in cache_map:
            final_results.append({
                "Model": model_id,
                "GPT_Ep": gpt_ep,
                "SoVITS_Ep": sovits_ep,
                "Reference": ref_id,
                "Score": cache_map[(model_id, ref_id)]
            })
            continue

        ref_path = os.path.join(REF_AUDIO_DIR, f"{ref_id}{REF_AUDIO_SUFFIX}.wav")
        deg_path = os.path.join(GEN_AUDIO_DIR, f)

        if os.path.exists(ref_path):
            tasks_to_run.append({
                "model_id": model_id,
                "gpt_ep": gpt_ep,
                "sovits_ep": sovits_ep,
                "ref_id": ref_id,
                "ref_path": ref_path,
                "deg_path": deg_path,
                "score": 0,
            })

    # ---- RUN EVALUATION ----
    if tasks_to_run:
        max_workers = min(MAX_THREADS, os.cpu_count() or 1)
        print(f"Processing {len(tasks_to_run)} new pairs with {max_workers} threads...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures: List[Future[TaskDict]] = [
                executor.submit(process_single_pair, t, VISQOL_MODEL_PATH)
                for t in tasks_to_run
            ]

            completed = 0

            for future in tqdm(as_completed(futures), total=len(tasks_to_run)):
                res = future.result()

                score = res["score"]
                if score is not None:
                    entry: ResultEntry = {
                        "Model": res["model_id"],
                        "GPT_Ep": res["gpt_ep"],
                        "SoVITS_Ep": res["sovits_ep"],
                        "Reference": res["ref_id"],
                        "Score": score,
                    }
                    final_results.append(entry)
                    cache_data.append(entry)

                    completed += 1
                    if completed % SAVE_INTERVAL == 0:
                        save_cache(CACHE_FILE, cache_data)

            save_cache(CACHE_FILE, cache_data)

    if not final_results:
        print("No results to analyze.")
        return

    # ---- ANALYZE ----
    df = pd.DataFrame(final_results)

    pivot_df = df.pivot(index="Model", columns="Reference", values="Score")
    pivot_df["Average"] = pivot_df.mean(axis=1) # type: ignore
    pivot_df = pivot_df.sort_index() # type: ignore

    gpt_means = df.groupby(['GPT_Ep', 'Reference'])['Score'].mean().unstack() # type: ignore
    if not gpt_means.empty:
        gpt_means["Average"] = gpt_means.mean(axis=1) # type: ignore
        gpt_means.index = [f"SUMMARY_GPT_{ep:03d}" for ep in gpt_means.index] # type: ignore

    sovits_means = df.groupby(['SoVITS_Ep', 'Reference'])['Score'].mean().unstack() # type: ignore
    if not sovits_means.empty:
        sovits_means["Average"] = sovits_means.mean(axis=1) # type: ignore
        sovits_means.index = [f"SUMMARY_SoVITS_{ep:03d}" for ep in sovits_means.index] # type: ignore

    final_display = pd.concat([pivot_df, gpt_means, sovits_means])

    print("\n" + "=" * 100)
    print("ViSQOL Evaluation Results (Individual Models & Aggregates)")
    print("=" * 100)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print(final_display)

    print("\n" + "=" * 50 + "\n📊 STATISTICAL ANALYSIS 📊\n" + "=" * 50)

    top_specific = analyze_distribution(pivot_df, "Specific Models")
    top_gpt = analyze_distribution(gpt_means, "GPT Epochs")
    top_sovits = analyze_distribution(sovits_means, "SoVITS Epochs")

    candidates_to_copy: Set[str] = set(top_specific.index.tolist())

    if not top_specific.empty and not top_gpt.empty and not top_sovits.empty:
        prefix = top_specific.index[0].split("_gpt")[0] # type: ignore

        gpt_eps = [int(x.split('_')[-1]) for x in top_gpt.index]
        sovits_eps = [int(x.split('_')[-1]) for x in top_sovits.index]

        print("\n" + "-" * 50)
        print("Generating 3x3 Matrix Candidates")
        print("-" * 50)

        for g in gpt_eps:
            for s in sovits_eps:
                model_name = f"{prefix}_gpt{g:03d}_sovits{s:03d}"
                candidates_to_copy.add(model_name)

    copy_candidates(candidates_to_copy, gen_files, GEN_AUDIO_DIR, CANDIDATES_DIR)


if __name__ == "__main__":
    main()
