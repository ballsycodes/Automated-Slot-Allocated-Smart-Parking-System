from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# CHANGE THIS PATH ONLY IF YOUR DATASET IS SOMEWHERE ELSE
# ============================================================

DATASET = Path(
    r"C:\Users\druva\final_project\Main_sampling_data\CNR-EXT-Patches-150x150"
)

PATCHES = DATASET / "PATCHES"
LABELS = DATASET / "LABELS"

TRAIN_FILE = LABELS / "train.txt"


# ============================================================
# READ TRAINING LABELS
# ============================================================

empty_images = []
occupied_images = []

with open(TRAIN_FILE, "r") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        image_path, label = line.rsplit(" ", 1)

        image_file = PATCHES / image_path

        if label == "0":
            empty_images.append(image_file)

        elif label == "1":
            occupied_images.append(image_file)


# ============================================================
# TAKE 6 RANDOM IMAGES FROM EACH CLASS
# ============================================================

random.seed(42)

empty_samples = random.sample(empty_images, 6)
occupied_samples = random.sample(occupied_images, 6)


# ============================================================
# DISPLAY EMPTY IMAGES
# ============================================================

fig, axes = plt.subplots(2, 6, figsize=(15, 6))

fig.suptitle("CNR-EXT Training Samples", fontsize=16)

for i, image_path in enumerate(empty_samples):

    image = Image.open(image_path)

    axes[0, i].imshow(image)
    axes[0, i].set_title("EMPTY (0)")
    axes[0, i].axis("off")

for i, image_path in enumerate(occupied_samples):

    image = Image.open(image_path)

    axes[1, i].imshow(image)
    axes[1, i].set_title("OCCUPIED (1)")
    axes[1, i].axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# CHECK IMAGE DIMENSIONS
# ============================================================

print("\nImage dimensions:")

for image_path in empty_samples[:3]:

    image = Image.open(image_path)

    print(image_path.name, "->", image.size)

for image_path in occupied_samples[:3]:

    image = Image.open(image_path)

    print(image_path.name, "->", image.size)