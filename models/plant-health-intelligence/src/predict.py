from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# PATHS
# ============================================================

MODULE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    MODULE_DIR
    / "saved_models"
    / "best_model_finetuned.keras"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
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
# TEST IMAGES
#
# Add a few images from different classes here.
# The parent folder is automatically treated as the actual label.
# ============================================================

TEST_IMAGES = [

    MODULE_DIR
    / "dataset"
    / "valid"
    / "Early_blight"
    / "0012b9d2-2130-4a06-a834-b1f3af34f57e___RS_Erly.B 8389.JPG",

    # Add more test images below.
    #
    # MODULE_DIR / "dataset" / "valid" / "healthy" / "image.jpg",
    #
    # MODULE_DIR / "dataset" / "valid" / "Late_blight" / "image.jpg",
    #
    # MODULE_DIR / "dataset" / "valid" / "Bacterial_spot" / "image.jpg",
]


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
# LOAD TRAINED MODEL
# ============================================================

print("Building Plant Health Intelligence model...")

model = build_model()

print("Architecture created.")

print("Loading trained weights...")

model.load_weights(
    MODEL_PATH
)

print("Trained weights loaded successfully.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_disease(image_path):

    image = keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    image = keras.utils.img_to_array(
        image
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    predictions = model.predict(
        image,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        predictions
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        predictions
    )


# ============================================================
# TEST SELECTED IMAGES
# ============================================================

print("\n" + "=" * 70)
print("PLANT HEALTH INTELLIGENCE - TEST PREDICTIONS")
print("=" * 70)


correct_predictions = 0
total_predictions = 0


for image_path in TEST_IMAGES:

    print("\n" + "-" * 70)

    print(
        f"Image: {image_path.name}"
    )

    # Check image exists
    if not image_path.exists():

        print(
            f"ERROR: Image not found:\n{image_path}"
        )

        continue

    # Actual class comes from the parent directory
    actual_class = image_path.parent.name

    # Predict
    (
        predicted_class,
        confidence,
        predictions
    ) = predict_disease(
        image_path
    )

    # Check correctness
    is_correct = (
        actual_class == predicted_class
    )

    total_predictions += 1

    if is_correct:
        correct_predictions += 1

    # Display result
    print(
        f"Actual Class : {actual_class}"
    )

    print(
        f"Prediction   : {predicted_class}"
    )

    print(
        f"Confidence   : "
        f"{confidence * 100:.2f}%"
    )

    if is_correct:

        print(
            "Result       : CORRECT"
        )

    else:

        print(
            "Result       : INCORRECT"
        )

    # Top 3 predictions
    top_indices = np.argsort(
        predictions
    )[-3:][::-1]

    print("\nTop 3 Predictions:")

    for index in top_indices:

        print(
            f"  {CLASS_NAMES[index]:45s}"
            f"{predictions[index] * 100:6.2f}%"
        )


# ============================================================
# TEST SUMMARY
# ============================================================

if total_predictions > 0:

    test_accuracy = (
        correct_predictions
        / total_predictions
        * 100
    )

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    print(
        f"Images Tested : {total_predictions}"
    )

    print(
        f"Correct       : {correct_predictions}"
    )

    print(
        f"Incorrect     : "
        f"{total_predictions - correct_predictions}"
    )

    print(
        f"Test Accuracy : "
        f"{test_accuracy:.2f}%"
    )

    print("=" * 70)

else:

    print(
        "\nNo valid test images were found."
    )