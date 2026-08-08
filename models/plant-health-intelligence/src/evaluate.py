from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ============================================================
# PATHS
# ============================================================

MODULE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = MODULE_DIR / "dataset"

VALID_DIR = DATASET_DIR / "valid"

MODEL_PATH = (
    MODULE_DIR
    / "saved_models"
    / "best_model_finetuned.keras"
)

OUTPUT_DIR = MODULE_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

NUM_CLASSES = 11


# ============================================================
# EXACT CLASS ORDER USED DURING TRAINING
# ============================================================

CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "healthy",
    "powdery_mildew"
]


# ============================================================
# CHECK PATHS
# ============================================================

if not VALID_DIR.exists():

    raise FileNotFoundError(
        f"Validation dataset not found:\n{VALID_DIR}"
    )


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3)
    )

    inputs = keras.Input(
        shape=(224, 224, 3)
    )

    x = base_model(
        inputs,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="PlantHealth_EfficientNetB0"
    )

    return model


# ============================================================
# START
# ============================================================

print("=" * 70)
print("PLANT HEALTH INTELLIGENCE")
print("AUTOMATED MODEL EVALUATION")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nBuilding model...")

model = build_model()

print("Architecture created.")

print("\nLoading trained weights...")

model.load_weights(
    MODEL_PATH
)

print("Trained weights loaded successfully.")


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...")

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,

    labels="inferred",

    label_mode="categorical",

    class_names=CLASS_NAMES,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False,

    color_mode="rgb"
)


# IMPORTANT:
# Get file count BEFORE prefetching because the
# PrefetchDataset object does not expose file_paths.

num_validation_images = len(
    valid_ds.file_paths
)


print("\nValidation dataset loaded.")

print(
    f"Number of validation images: "
    f"{num_validation_images}"
)

print(
    f"Number of classes: "
    f"{NUM_CLASSES}"
)


# Prefetch after reading file count

valid_ds = valid_ds.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)

print(
    "Starting prediction on entire validation dataset..."
)

print("=" * 70)

y_true = []

y_pred = []

processed = 0


for images, labels in valid_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    true_labels = np.argmax(
        labels.numpy(),
        axis=1
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        true_labels
    )

    y_pred.extend(
        predicted_labels
    )

    processed += images.shape[0]

    print(
        f"Processed "
        f"{processed}/{num_validation_images}",
        end="\r"
    )


y_true = np.array(
    y_true
)

y_pred = np.array(
    y_pred
)


print(
    "\n\nPrediction completed."
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


print("\n" + "=" * 70)
print("OVERALL RESULT")
print("=" * 70)

print(
    f"Validation Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
)


print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# Save report

report_path = (
    OUTPUT_DIR
    / "classification_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PLANT HEALTH INTELLIGENCE\n"
    )

    file.write(
        "=========================\n\n"
    )

    file.write(
        f"Validation Images: "
        f"{num_validation_images}\n"
    )

    file.write(
        f"Validation Accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    file.write(
        report
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


plt.figure(
    figsize=(14, 12)
)


plt.imshow(
    cm,
    interpolation="nearest"
)


plt.title(
    "Plant Health Intelligence - Confusion Matrix"
)


plt.colorbar()


tick_marks = np.arange(
    NUM_CLASSES
)


plt.xticks(
    tick_marks,
    CLASS_NAMES,
    rotation=90
)


plt.yticks(
    tick_marks,
    CLASS_NAMES
)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "Actual Class"
)


# Add values inside matrix

for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()


# Save confusion matrix

confusion_matrix_path = (
    OUTPUT_DIR
    / "confusion_matrix.png"
)


plt.savefig(
    confusion_matrix_path,
    dpi=200,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"\nValidation Images : "
    f"{num_validation_images}"
)

print(
    f"Validation Accuracy : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"\nClassification report saved to:"
    f"\n{report_path}"
)

print(
    f"\nConfusion matrix saved to:"
    f"\n{confusion_matrix_path}"
)

print("=" * 70)