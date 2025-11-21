#/usr/bin/env python

# put it in GPT-SoVITS project root path

# Usage example
#  python evaluation_batch_inference.py --ref /data/kotomi-evaluation/z4421_01700 --input /data/kotomi-evaluation/input --output /data/kotomi-evaluation/output --version v4 --project kotomi --gpt-epoch "1-100"   --sovits-epoch "1-100"

import os
import sys
import re
import glob
import argparse
from argparse import Namespace
from typing import Set, Optional, List, Dict, Final, Tuple, Any

import soundfile as sf
from tqdm import tqdm

script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# Add current directory to sys.path to ensure imports work
sys.path.append(os.getcwd())

# Import necessary modules from the existing GPT-SoVITS project
try:
    # Import i18n from webui to ensure we use the exact same translation logic/state
    from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav, i18n # type: ignore
except ImportError:
    print("Error: Could not import GPT_SoVITS modules. Please ensure you are running this script from the GPT-SoVITS project root.")
    sys.exit(1)

# Language mapping for CLI convenience
# We wrap strings in i18n() so they match the keys in inference_webui's dict_language
LANG_MAP: Dict[str, Any] = {
    "ja": i18n("日文"),
    "zh": i18n("中文"),
    "en": i18n("英文"),
    "mix": i18n("中英混合"), 
    "auto": i18n("多语种混合")
}

# Version to Directory Mapping
VERSION_DIRS: Final[Dict[str, Tuple[str,str]]] = {
    "v2": ("GPT_weights_v2", "SoVITS_weights_v2"),
    "v2pro": ("GPT_weights_v2Pro", "SoVITS_weights_v2Pro"),
    "v2proplus": ("GPT_weights_v2ProPlus", "SoVITS_weights_v2ProPlus"),
    "v3": ("GPT_weights_v3", "SoVITS_weights_v3"),
    "v4": ("GPT_weights_v4", "SoVITS_weights_v4"),
}

VERSION_SAMPLE_STEPS: Final[Dict[str, int]] = {
    "v2": 8,
    "v2pro": 8,
    "v2proplus": 8,
    "v3": 32, # max: 128
    "v4": 32, # max: 32
}

def parse_epoch_range(range_str: Optional[str]) -> Optional[Set[int]]:
    """
    Parses a string like "20,30,40-45" into a set of integers {20, 30, 40, 41, ..., 45}.
    Returns None if input is None or empty, implying 'all epochs'.
    """
    if not range_str:
        return None
    
    epochs: Set[int] = set()
    try:
        parts = range_str.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                epochs.update(range(start, end + 1))
            else:
                epochs.add(int(part))
    except ValueError:
        print(f"Error parsing epoch range: '{range_str}'. Format should be '10,20-30'.")
        sys.exit(1)
        
    return epochs

def get_epoch_from_filename(filename:str):
    """
    Extracts epoch number from filenames like:
    kotomi-e100.ckpt -> 100
    kotomi_e100_s31400.pth -> 100
    """
    match = re.search(r'[-_]e(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def load_model_list(base_dir:str, project_prefix:str, extension:str, allowed_epochs: Optional[Set[int]] = None) -> List[str]:
    """
    Lists and sorts model files by epoch.
    Safe check: Ensures filename starts with 'prefix-' or 'prefix_'
    to avoid matching 'test' with 'testing'.
    Filters by allowed_epochs if provided.
    """
    if not os.path.exists(base_dir):
        print(f"Warning: Directory {base_dir} does not exist.")
        return []
    
    # Define valid prefixes (e.g., "kotomi-" and "kotomi_")
    valid_prefixes = (f"{project_prefix}-", f"{project_prefix}_")
    
    files:list[str] = []
    for f in os.listdir(base_dir):
        if f.startswith(valid_prefixes) and f.endswith(extension):
            # Filter logic
            if allowed_epochs is not None:
                epoch = get_epoch_from_filename(f)
                if epoch not in allowed_epochs:
                    continue
            files.append(f)
    
    # Sort by epoch number
    files.sort(key=get_epoch_from_filename)
    return [os.path.join(base_dir, f) for f in files]

def run_inference(args: Namespace):
    # 1. Setup Directories and Paths
    if args.version not in VERSION_DIRS:
        print(f"Error: Version {args.version} not supported. Choices: {list(VERSION_DIRS.keys())}")
        return

    gpt_dir_name, sovits_dir_name = VERSION_DIRS[args.version]
    gpt_dir = os.path.join(os.getcwd(), gpt_dir_name)
    sovits_dir = os.path.join(os.getcwd(), sovits_dir_name)
    
    os.makedirs(args.output, exist_ok=True)

    # 2. Parse Epoch Filters
    gpt_allowed_epochs = parse_epoch_range(args.gpt_epoch)
    sovits_allowed_epochs = parse_epoch_range(args.sovits_epoch)

    # 3. Load Reference Data
    ref_wav_path = args.ref + ".wav"
    ref_txt_path = args.ref + ".txt"
    
    if not os.path.exists(ref_wav_path) or not os.path.exists(ref_txt_path):
        print(f"Error: Reference files not found: {ref_wav_path} or {ref_txt_path}")
        return

    with open(ref_txt_path, "r", encoding="utf-8") as f:
        ref_text = f.read().strip()
    
    # 4. Load Input Texts
    input_files = glob.glob(os.path.join(args.input, "*.txt"))
    if not input_files:
        print(f"Error: No .txt files found in {args.input}")
        return
    
    print(f"Found {len(input_files)} input text files.")

    # 5. Get Model Lists
    gpt_models = load_model_list(gpt_dir, args.project, ".ckpt", allowed_epochs=gpt_allowed_epochs)
    sovits_models = load_model_list(sovits_dir, args.project, ".pth", allowed_epochs=sovits_allowed_epochs)

    print(f"Found {len(gpt_models)} GPT models in {gpt_dir}")
    print(f"Found {len(sovits_models)} SoVITS models in {sovits_dir}")

    if not gpt_models or not sovits_models:
        print("Error: Missing models. Check project prefix, directory, or epoch filters.")
        return

    target_lang_ui = LANG_MAP.get(args.lang, i18n("日文"))  # type: ignore
    ref_lang_ui = LANG_MAP.get(args.lang, i18n("日文")) # type: ignore

    # 6. Batch Processing Loop
    total_ops = len(gpt_models) * len(sovits_models) * len(input_files)
    pbar = tqdm(total=total_ops, desc="Batch Inference")

    for gpt_path in gpt_models:
        gpt_epoch = get_epoch_from_filename(os.path.basename(gpt_path))
        if gpt_epoch < 1 or gpt_epoch > 999:
            print(f"Ignore invalid gpt epoch {gpt_path}")
            continue
        
        try:
            change_gpt_weights(gpt_path=gpt_path)
        except Exception as e:
            print(f"Failed to load GPT: {gpt_path} - {e}")
            continue

        for sovits_path in sovits_models:
            sovits_epoch = get_epoch_from_filename(os.path.basename(sovits_path))
            if sovits_epoch < 1 or sovits_epoch > 999:
                print(f"Ignore invalid sovits epoch {sovits_path}")
                continue
            
            try:
                generator = change_sovits_weights(sovits_path=sovits_path, prompt_language=ref_lang_ui, text_language=target_lang_ui) # type: ignore
                next(generator) # type: ignore
            except StopIteration:
                pass
            except Exception as e:
                print(f"Failed to load SoVITS: {sovits_path} - {e}")
                continue

            for input_txt_path in input_files:
                input_filename = os.path.splitext(os.path.basename(input_txt_path))[0]
                
                # Format: 3-digit epoch number for better sorting (015, 050, 100)
                output_fname = f"{args.project}_{args.version}_gpt{gpt_epoch:03d}_sovits{sovits_epoch:03d}_{input_filename}.wav"
                output_full_path = os.path.join(args.output, output_fname)

                if os.path.exists(output_full_path):
                    pbar.update(1)
                    continue

                with open(input_txt_path, "r", encoding="utf-8") as f:
                    target_text = f.read().strip()

                temp_output_path = output_full_path + ".tmp"
                try:
                    synthesis_result = get_tts_wav( # type: ignore
                        ref_wav_path=ref_wav_path,
                        prompt_text=ref_text,
                        prompt_language=ref_lang_ui,
                        text=target_text,
                        text_language=target_lang_ui,
                        top_p=1,
                        temperature=1,
                        top_k=15,
                        sample_steps=VERSION_SAMPLE_STEPS[args.version],
                        how_to_cut=i18n("凑四句一切"),
                    )
                    
                    result_list = list(synthesis_result) # type: ignore
                    if result_list:
                        last_sampling_rate, last_audio_data = result_list[-1] # type: ignore
                        
                        # --- ATOMIC WRITE START ---
                        sf.write(temp_output_path, last_audio_data, last_sampling_rate, format='WAV') # type: ignore
                        if os.path.exists(output_full_path):
                            os.remove(temp_output_path)
                        else:
                            os.rename(temp_output_path, output_full_path)
                        # --- ATOMIC WRITE END ---
                
                except Exception as e:
                    # Print detailed error for debugging
                    print(f"\nError generating {output_fname}: {e}")
                    # Clean up temp file
                    if 'temp_output_path' in locals() and os.path.exists(temp_output_path):
                        try:
                            os.remove(temp_output_path)
                        except:
                            pass

                pbar.update(1)

    pbar.close()
    print("Batch generation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS Evaluation Batch Inference")
    
    parser.add_argument("--ref", required=True, help="Path to reference file base name (without extension). Expects .wav and .txt exist.")
    parser.add_argument("--input", required=True, help="Directory containing input .txt files")
    parser.add_argument("--output", required=True, help="Directory to save output .wav files")
    parser.add_argument("--version", required=True, choices=["v2", "v2pro", "v2proplus", "v3", "v4"], help="Model version to determine weight directories")
    parser.add_argument("--project", required=True, help="Project prefix (e.g., kotomi) to filter models")
    parser.add_argument("--lang", default="ja", choices=["ja", "zh", "en", "mix", "auto"], help="Target language (default: ja)")
    
    # New arguments for fine-grained control
    parser.add_argument("--gpt-epoch", help="Filter GPT epochs (e.g., '10,20-30'). If omitted, uses all.")
    parser.add_argument("--sovits-epoch", help="Filter SoVITS epochs (e.g., '50,60-100'). If omitted, uses all.")

    args = parser.parse_args()
    
    run_inference(args)