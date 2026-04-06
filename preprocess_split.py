from ase.db import connect
import os
import random
import math
import csv
from collections import defaultdict, Counter

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required. Install with: pip3 install torch")

# =========================
# Settings
# =========================
DB_PATH = "mof_plus_adsorbate/part_00000.aselmdb"
OUT_DIR = "processed_data"
SEED = 42

# Split ratios by unique MOF name
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Save this many samples per file
CHUNK_SIZE = 25000

# Filter choice:
# True  -> only clean single-adsorbate CO2 rows
# False -> any row with CO2 present
STRICT_SINGLE_CO2 = True


def is_co2_row(row_data: dict) -> bool:
    nco2 = row_data.get("nco2", 0)
    if nco2 <= 0:
        return False

    if not STRICT_SINGLE_CO2:
        return True

    return (
        row_data.get("nco2", 0) > 0
        and row_data.get("nh2o", 0) == 0
        and row_data.get("nn2", 0) == 0
        and row_data.get("no2", 0) == 0
        and row_data.get("nads", 0) == row_data.get("nco2", 0)
    )


def save_chunk(samples, split_name, chunk_idx):
    if not samples:
        return None

    split_dir = os.path.join(OUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)

    out_path = os.path.join(split_dir, f"{split_name}_part_{chunk_idx:03d}.pt")
    torch.save(samples, out_path)
    return out_path


def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Opening database: {DB_PATH}")
    db = connect(DB_PATH)

    # -------------------------
    # Pass 1: collect unique MOF names that have CO2 rows
    # -------------------------
    co2_mofs = set()
    total_co2_rows = 0

    print("Pass 1: scanning for CO2 rows and MOF names...")
    for row in db.select():
        row_data = row.data if row.data is not None else {}
        if is_co2_row(row_data):
            mof_name = row_data.get("mof_name")
            if mof_name is None:
                continue
            co2_mofs.add(mof_name)
            total_co2_rows += 1

    co2_mofs = sorted(co2_mofs)
    random.shuffle(co2_mofs)

    n_mofs = len(co2_mofs)
    n_train = int(n_mofs * TRAIN_RATIO)
    n_val = int(n_mofs * VAL_RATIO)
    n_test = n_mofs - n_train - n_val

    train_mofs = set(co2_mofs[:n_train])
    val_mofs = set(co2_mofs[n_train:n_train + n_val])
    test_mofs = set(co2_mofs[n_train + n_val:])

    print("\n=== MOF split summary ===")
    print(f"Unique CO2 MOFs: {n_mofs}")
    print(f"Train MOFs: {len(train_mofs)}")
    print(f"Val MOFs:   {len(val_mofs)}")
    print(f"Test MOFs:  {len(test_mofs)}")
    print(f"Total CO2 rows found: {total_co2_rows}")

    # Save MOF split list
    mof_split_csv = os.path.join(OUT_DIR, "mof_split.csv")
    with open(mof_split_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mof_name", "split"])
        for m in sorted(train_mofs):
            writer.writerow([m, "train"])
        for m in sorted(val_mofs):
            writer.writerow([m, "val"])
        for m in sorted(test_mofs):
            writer.writerow([m, "test"])

    print(f"Saved MOF split file: {mof_split_csv}")

    # -------------------------
    # Pass 2: write chunked samples to train/val/test
    # -------------------------
    split_counters = Counter()
    chunk_counters = {"train": 0, "val": 0, "test": 0}
    buffers = {"train": [], "val": [], "test": []}

    metadata_csv = os.path.join(OUT_DIR, "sample_metadata.csv")
    meta_f = open(metadata_csv, "w", newline="")
    meta_writer = csv.writer(meta_f)
    meta_writer.writerow(
        [
            "split",
            "chunk_file",
            "mof_name",
            "sample_name",
            "energy_ads_corrected",
            "n_atoms",
        ]
    )

    print("\nPass 2: converting rows and saving chunks...")
    for row in db.select():
        row_data = row.data if row.data is not None else {}
        if not is_co2_row(row_data):
            continue

        mof_name = row_data.get("mof_name")
        if mof_name in train_mofs:
            split = "train"
        elif mof_name in val_mofs:
            split = "val"
        elif mof_name in test_mofs:
            split = "test"
        else:
            continue

        atoms = row.toatoms()

        sample = {
            "pos": torch.tensor(atoms.positions, dtype=torch.float32),
            "z": torch.tensor(atoms.numbers, dtype=torch.long),
            "y": torch.tensor(float(row_data["energy_ads_corrected"]), dtype=torch.float32),
            "mof_name": mof_name,
            "sample_name": row_data.get("name", ""),
            "nco2": int(row_data.get("nco2", 0)),
            "nh2o": int(row_data.get("nh2o", 0)),
            "nn2": int(row_data.get("nn2", 0)),
            "no2": int(row_data.get("no2", 0)),
            "nads": int(row_data.get("nads", 0)),
            "n_atoms": len(atoms),
            "sid": row_data.get("sid"),
            "fid": row_data.get("fid"),
        }

        buffers[split].append(sample)
        split_counters[split] += 1

        if len(buffers[split]) >= CHUNK_SIZE:
            out_path = save_chunk(buffers[split], split, chunk_counters[split])
            for s in buffers[split]:
                meta_writer.writerow([
                    split,
                    os.path.basename(out_path),
                    s["mof_name"],
                    s["sample_name"],
                    float(s["y"]),
                    s["n_atoms"],
                ])
            print(f"Saved {out_path} with {len(buffers[split])} samples")
            buffers[split] = []
            chunk_counters[split] += 1

    # Save leftovers
    for split in ["train", "val", "test"]:
        if buffers[split]:
            out_path = save_chunk(buffers[split], split, chunk_counters[split])
            for s in buffers[split]:
                meta_writer.writerow([
                    split,
                    os.path.basename(out_path),
                    s["mof_name"],
                    s["sample_name"],
                    float(s["y"]),
                    s["n_atoms"],
                ])
            print(f"Saved {out_path} with {len(buffers[split])} samples")

    meta_f.close()

    print("\n=== Final row summary ===")
    print(f"Train rows: {split_counters['train']}")
    print(f"Val rows:   {split_counters['val']}")
    print(f"Test rows:  {split_counters['test']}")
    print(f"Metadata CSV saved to: {metadata_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()


