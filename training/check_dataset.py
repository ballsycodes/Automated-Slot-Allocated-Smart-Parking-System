from pathlib import Path

# Change this to wherever you extracted the dataset
DATASET = Path(r"C:\Users\druva\final_project\Main_sampling_data\CNR-EXT-Patches-150x150")

PATCHES = DATASET / "PATCHES"
LABELS = DATASET / "LABELS"

for split in ["train", "val", "test"]:
    label_file = LABELS / f"{split}.txt"

    print(f"\n===== {split.upper()} =====")

    if not label_file.exists():
        print("Label file not found:", label_file)
        continue

    total = 0
    free = 0
    busy = 0
    missing = 0

    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            image_path, label = line.rsplit(" ", 1)

            image_file = PATCHES / image_path

            total += 1

            if label == "0":
                free += 1
            elif label == "1":
                busy += 1

            if not image_file.exists():
                missing += 1

    print("Total images :", total)
    print("Free (0)     :", free)
    print("Busy (1)     :", busy)
    print("Missing files:", missing)