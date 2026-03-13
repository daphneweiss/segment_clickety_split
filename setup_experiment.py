#!/usr/bin/env python3
"""
Setup script: generates per-condition stimulus lists and naming config
from the master CSV.

Naming convention for critical items:
  {speaker}_{stem}_{stim_type}-{token_num}.wav
  e.g., f1_dino_crit_s-2.wav, f1_dino_crit_s_sh-1.wav

  The stem is the letters before the fricative (extracted from the
  manipulated version: dino(SH)aur → stem = "dino").

  Best/final tokens drop the suffix: f1_dino_crit_s.wav

For fillers:
  {speaker}_{word}_{stim_type}-{token_num}.wav
  e.g., f1_hamburger_fill_word-1.wav

Usage:
  python setup_experiment.py \
      --csv all_words_concatenated.csv \
      --speaker_dirs f1/ f2/ m1/ m2/ \
      --output_dir experiment/
"""

import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict
import re


# Map stim_type → recording session folder name
TYPE_TO_FOLDER = {
    "crit_sh":    "critical_sh_normal",
    "crit_sh_s":  "critical_sh_s",
    "crit_s":     "critical_s_normal",
    "crit_s_sh":  "critical_s_sh",
    "fill_non":   "filler_pseudo",
    "fill_word":  "filler_word",
}


def extract_stem(word):
    """Extract stem from a manipulated word: dino(SH)aur → dino, ambi(S)on → ambi."""
    if "(S)" in word or "(SH)" in word:
        return word.split("(")[0].lower()
    return None


def build_naming_config(csv_path):
    """
    Build the naming config: for each word, determine the filename base.

    Returns:
      by_folder: dict[folder_name] → list of {word, stim_type, file_stem} in recording order
      pair_map:  dict[stem] → {crit_s: word, crit_s_sh: word} or {crit_sh: word, crit_sh_s: word}
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["stim_type"]].append(r["words"])

    # Build stem map from the manipulated versions
    # crit_sh_s words have (S) → stem pairs with crit_sh
    # crit_s_sh words have (SH) → stem pairs with crit_s
    stem_map = {}  # manipulated_word → stem

    for word in by_type.get("crit_sh_s", []):
        stem_map[word] = extract_stem(word)
    for word in by_type.get("crit_s_sh", []):
        stem_map[word] = extract_stem(word)

    # Match pairs by position (same index in each list)
    pair_map = {}  # stem → {type: word, ...}

    # crit_sh ↔ crit_sh_s
    for i, (normal, manip) in enumerate(zip(by_type.get("crit_sh", []),
                                             by_type.get("crit_sh_s", []))):
        stem = stem_map.get(manip, normal.lower()[:4])
        pair_map[stem] = {"crit_sh": normal, "crit_sh_s": manip}

    # crit_s ↔ crit_s_sh
    for i, (normal, manip) in enumerate(zip(by_type.get("crit_s", []),
                                             by_type.get("crit_s_sh", []))):
        stem = stem_map.get(manip, normal.lower()[:4])
        pair_map[stem] = pair_map.get(stem, {})
        pair_map[stem].update({"crit_s": normal, "crit_s_sh": manip})

    # Build per-folder config with file_stem for each word
    # Reverse lookup: word → stem
    word_to_stem = {}
    for stem, types in pair_map.items():
        for stype, word in types.items():
            word_to_stem[word] = stem

    by_folder = defaultdict(list)
    for r in rows:
        word, stype = r["words"], r["stim_type"]
        folder = TYPE_TO_FOLDER.get(stype, stype)

        if word in word_to_stem:
            file_stem = f"{word_to_stem[word]}_{stype}"
        else:
            # Fillers: use the word itself (lowercase, sanitized)
            safe = re.sub(r'[^\w]', '', word.lower())
            file_stem = f"{safe}_{stype}"

        by_folder[folder].append({
            "word": word,
            "stim_type": stype,
            "file_stem": file_stem,
        })

    return by_folder, pair_map, word_to_stem


def setup_experiment(csv_path, speaker_dirs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_folder, pair_map, word_to_stem = build_naming_config(csv_path)

    print("=" * 60)
    print("EXPERIMENT SETUP")
    print("=" * 60)

    # Save master naming config
    config = {
        "pair_map": pair_map,
        "word_to_stem": word_to_stem,
        "folders": {folder: items for folder, items in by_folder.items()},
    }
    config_path = output_dir / "naming_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Naming config: {config_path}")

    # Print pair summary
    print(f"\n  Critical pairs ({len(pair_map)}):")
    for stem, types in sorted(pair_map.items()):
        parts = [f"{t}={w}" for t, w in sorted(types.items())]
        print(f"    {stem}: {', '.join(parts)}")

    # Print folder summary
    print(f"\n  Recording folders ({len(by_folder)}):")
    for folder, items in sorted(by_folder.items()):
        print(f"    {folder}: {len(items)} words")

    # Generate per-folder stimuli.txt files
    # These go INTO each speaker's recording directories
    stim_lists_dir = output_dir / "stimulus_lists"
    stim_lists_dir.mkdir(exist_ok=True)

    for folder, items in by_folder.items():
        stim_path = stim_lists_dir / f"{folder}_stimuli.txt"
        with open(stim_path, "w") as f:
            f.write(f"# Stimulus list for {folder}\n")
            f.write(f"# {len(items)} items in recording order\n")
            for item in items:
                f.write(f"{item['file_stem']}\n")
        print(f"    {stim_path.name}")

    # Generate interleaved lists for anomalous sessions where a speaker
    # recorded paired conditions together (e.g., each crit_s word immediately
    # followed by its crit_s_sh counterpart in the same recording).
    interleave_combos = [
        ("critical_s_normal", "critical_s_sh"),
        ("critical_sh_normal", "critical_sh_s"),
    ]
    for folder_a, folder_b in interleave_combos:
        if folder_a not in by_folder or folder_b not in by_folder:
            continue
        items_a = by_folder[folder_a]
        items_b = by_folder[folder_b]
        if len(items_a) != len(items_b):
            print(f"    ⚠ Cannot interleave {folder_a} ({len(items_a)}) "
                  f"and {folder_b} ({len(items_b)}): different lengths")
            continue

        interleaved_name = f"{folder_a}+{folder_b}_interleaved_stimuli.txt"
        stim_path = stim_lists_dir / interleaved_name
        with open(stim_path, "w") as f:
            f.write(f"# Interleaved list: {folder_a} + {folder_b}\n")
            f.write(f"# For sessions where each normal word was immediately\n")
            f.write(f"# followed by its manipulated counterpart\n")
            f.write(f"# {len(items_a) + len(items_b)} items total\n")
            for a, b in zip(items_a, items_b):
                f.write(f"{a['file_stem']}\n")
                f.write(f"{b['file_stem']}\n")
        print(f"    {interleaved_name} (anomalous session order)")

    # Generate per-speaker run scripts
    print(f"\n  Per-speaker setup:")
    for spk_dir in speaker_dirs:
        spk_dir = Path(spk_dir)
        spk_id = spk_dir.stem  # e.g., "f1" from "f1/"

        # Copy stimulus lists into each condition folder
        for folder in by_folder:
            cond_dir = spk_dir / folder
            if cond_dir.exists():
                src = stim_lists_dir / f"{folder}_stimuli.txt"
                dst = cond_dir / "stimuli.txt"
                if not dst.exists():
                    import shutil
                    shutil.copy2(src, dst)
                    print(f"    Copied stimuli.txt → {dst}")
                else:
                    print(f"    {dst} already exists, skipping")
            else:
                print(f"    ⚠ {cond_dir} does not exist")

    # Generate the continuum pairing script
    pairs_path = output_dir / "continuum_pairs.json"
    continuum_pairs = []
    for stem, types in pair_map.items():
        # /s/ words pair with /s_sh/ manipulated versions
        if "crit_s" in types and "crit_s_sh" in types:
            continuum_pairs.append({
                "stem": stem,
                "s_condition": "critical_s_normal",
                "sh_condition": "critical_s_sh",
                "s_file_stem": f"{stem}_crit_s",
                "sh_file_stem": f"{stem}_crit_s_sh",
                "s_word": types["crit_s"],
                "sh_word": types["crit_s_sh"],
            })
        # /sh/ words pair with /sh_s/ manipulated versions
        if "crit_sh" in types and "crit_sh_s" in types:
            continuum_pairs.append({
                "stem": stem,
                "s_condition": "critical_sh_s",
                "sh_condition": "critical_sh_normal",
                "s_file_stem": f"{stem}_crit_sh_s",
                "sh_file_stem": f"{stem}_crit_sh",
                "s_word": types["crit_sh_s"],
                "sh_word": types["crit_sh"],
            })

    with open(pairs_path, "w") as f:
        json.dump(continuum_pairs, f, indent=2)
    print(f"\n  Continuum pairs ({len(continuum_pairs)}): {pairs_path}")

    print(f"\n{'=' * 60}")
    print(f"DONE. See {output_dir}/README_run.md for the full pipeline.")
    print(f"{'=' * 60}")

    return config, continuum_pairs


def main():
    parser = argparse.ArgumentParser(description="Set up experiment from stimulus CSV.")
    parser.add_argument("--csv", required=True, help="Master stimulus CSV")
    parser.add_argument("--speaker_dirs", nargs="+", default=[],
                        help="Speaker recording directories (e.g., f1/ f2/ m1/ m2/)")
    parser.add_argument("--output_dir", default="experiment", help="Output directory")
    args = parser.parse_args()

    setup_experiment(args.csv, args.speaker_dirs, args.output_dir)


if __name__ == "__main__":
    main()
