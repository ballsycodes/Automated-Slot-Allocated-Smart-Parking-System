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
VAL_FILE = LABELS / "val.txt"
TEST_FILE = LABELS / "test.txt"

IMAGE_SIZE = (96, 96)
BATCH_SIZE = 32

# Initially use only 20,000 training images
SAMPLES_PER_CLASS = 10_000

EPOCHS = 20
SEED = 42

random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# READ LABEL FILE
# ============================================================

def read_labels(label_file):

    empty = []
    occupied = []

    with open(label_file, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            image_path, label = line.rsplit(" ", 1)

            image_file = str(PATCHES / image_path)

            if label == "0":
                empty.append(image_file)

            elif label == "1":
                occupied.append(image_file)

    return empty, occupied


# ============================================================
# LOAD TRAINING DATA
# ============================================================

train_empty, train_occupied = read_labels(TRAIN_FILE)

print("Available training images:")
print("Empty    :", len(train_empty))
print("Occupied :", len(train_occupied))


# Balanced 20k development set

train_empty = random.sample(
    train_empty,
    SAMPLES_PER_CLASS
)

train_occupied = random.sample(
    train_occupied,
    SAMPLES_PER_CLASS
)

train_paths = train_empty + train_occupied

train_labels = (
    [0] * len(train_empty)
    +
    [1] * len(train_occupied)
)

combined = list(zip(train_paths, train_labels))

random.shuffle(combined)

train_paths, train_labels = zip(*combined)

train_paths = list(train_paths)
train_labels = list(train_labels)


print("\nDevelopment training set:")
print("Total    :", len(train_paths))
print("Empty    :", train_labels.count(0))
print("Occupied :", train_labels.count(1))


# ============================================================
# LOAD VALIDATION + TEST DATA
# ============================================================

val_empty, val_occupied = read_labels(VAL_FILE)
test_empty, test_occupied = read_labels(TEST_FILE)

val_paths = val_empty + val_occupied
val_labels = (
    [0] * len(val_empty)
    +
    [1] * len(val_occupied)
)

test_paths = test_empty + test_occupied
test_labels = (
    [0] * len(test_empty)
    +
    [1] * len(test_occupied)
)

print("\nValidation:")
print("Total    :", len(val_paths))
print("Empty    :", val_labels.count(0))
print("Occupied :", val_labels.count(1))

print("\nTest:")
print("Total    :", len(test_paths))
print("Empty    :", test_labels.count(0))
print("Occupied :", test_labels.count(1))


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image, tf.cast(label, tf.int32)


# ============================================================
# CREATE DATASETS
# ============================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

test_ds = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)


# Training shuffle

train_ds = train_ds.shuffle(
    len(train_paths),
    seed=SEED
)


# Load images

train_ds = train_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
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


def augment(image, label):

    image = augmentation(
        image,
        training=True
    )

    return image, label


train_ds = train_ds.map(
    augment,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# BATCH + PREFETCH
# ============================================================

train_ds = train_ds.batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

val_ds = val_ds.batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

test_ds = test_ds.batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# BUILD CNN
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(96, 96, 3)
    ),

    tf.keras.layers.Conv2D(
        16,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.3
    ),

    tf.keras.layers.Dense(
        2,
        activation="softmax"
    )
])


# ============================================================
# DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "parking_cnn_96.keras",
        monitor="val_loss",
        save_best_only=True
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=callbacks
)


# ============================================================
# TEST
# ============================================================

print("\nEvaluating on TEST set...\n")

test_loss, test_accuracy = model.evaluate(
    test_ds
)

print("\n================================")
print("FINAL TEST RESULTS")
print("================================")

print("Test loss     :", test_loss)
print("Test accuracy :", test_accuracy)


# ============================================================
# TRAINING GRAPHS
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "CNR-EXT CNN Training Accuracy"
)

plt.legend()

plt.grid()

plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "CNR-EXT CNN Training Loss"
)

plt.legend()

plt.grid()

plt.show()