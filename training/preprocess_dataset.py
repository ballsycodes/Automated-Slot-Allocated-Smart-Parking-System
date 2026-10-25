from pathlib import Path
import random
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

DATASET = Path(
    r"C:\Users\druva\final_project\Main_sampling_data\CNR-EXT-Patches-150x150"
)

PATCHES = DATASET / "PATCHES"
LABELS = DATASET / "LABELS"

TRAIN_FILE = LABELS / "train.txt"

# Development subset
SAMPLES_PER_CLASS = 10_000

# First baseline resolution
IMAGE_SIZE = (150, 150)

# Number of images processed together
BATCH_SIZE = 32

# Reproducibility
SEED = 42


# ============================================================
# 1. READ LABEL FILE
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
            empty_images.append(str(image_file))

        elif label == "1":
            occupied_images.append(str(image_file))


print("Available training images:")
print("Empty     :", len(empty_images))
print("Occupied  :", len(occupied_images))


# ============================================================
# 2. CREATE BALANCED DEVELOPMENT SUBSET
# ============================================================

random.seed(SEED)

empty_samples = random.sample(
    empty_images,
    SAMPLES_PER_CLASS
)

occupied_samples = random.sample(
    occupied_images,
    SAMPLES_PER_CLASS
)


# Combine them

image_paths = empty_samples + occupied_samples

labels = (
    [0] * len(empty_samples)
    +
    [1] * len(occupied_samples)
)


# Shuffle images and labels together

combined = list(zip(image_paths, labels))

random.shuffle(combined)

image_paths, labels = zip(*combined)

image_paths = list(image_paths)
labels = list(labels)


print("\nDevelopment dataset:")
print("Total     :", len(image_paths))
print("Empty     :", labels.count(0))
print("Occupied  :", labels.count(1))


# ============================================================
# 3. LOAD IMAGE
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    # First experiment keeps original 150x150 resolution
    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    # Convert 0-255 → 0-1
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# ============================================================
# 4. CREATE BASE TENSORFLOW DATASET
# ============================================================

dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths, labels)
)

dataset = dataset.shuffle(
    buffer_size=len(image_paths),
    seed=SEED,
    reshuffle_each_iteration=True
)

dataset = dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# 5. DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential([
    
    tf.keras.layers.RandomRotation(
        factor=0.03
    ),

    tf.keras.layers.RandomTranslation(
        height_factor=0.03,
        width_factor=0.03
    ),

    tf.keras.layers.RandomZoom(
        height_factor=0.03,
        width_factor=0.03
    ),

    tf.keras.layers.RandomBrightness(
        factor=0.10,
        value_range=(0, 1)
    )
])


# ============================================================
# 6. APPLY AUGMENTATION ONLY TO TRAINING PIPELINE
# ============================================================

def augment_image(image, label):

    image = augmentation(
        image,
        training=True
    )

    return image, label


dataset_augmented = dataset.map(
    augment_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# 7. BATCH + PREFETCH
# ============================================================

dataset_augmented = dataset_augmented.batch(
    BATCH_SIZE
)

dataset_augmented = dataset_augmented.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 8. DISPLAY AUGMENTED SAMPLES
# ============================================================

images, labels_batch = next(
    iter(dataset_augmented)
)

plt.figure(figsize=(12, 8))

for i in range(12):

    plt.subplot(3, 4, i + 1)

    plt.imshow(images[i].numpy())

    label_name = (
        "EMPTY"
        if labels_batch[i] == 0
        else "OCCUPIED"
    )

    plt.title(label_name)

    plt.axis("off")


plt.suptitle(
    "Augmented Training Samples",
    fontsize=16
)

plt.tight_layout()

plt.show()


# ============================================================
# 9. FINAL INFORMATION
# ============================================================

print("\nDataset pipeline created successfully.")

print("Image size :", IMAGE_SIZE)
print("Batch size :", BATCH_SIZE)

print("\nPipeline:")
print("150x150 image")
print("     ↓")
print("Resize")
print("     ↓")
print("Normalize 0-1")
print("     ↓")
print("Augmentation")
print("     ↓")
print("CNN")