import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import sys 

# checks to see if you provided a name, e.g., "python modeltrain.py baseline"
if len(sys.argv) > 1:
    EXPERIMENT_NAME = sys.argv[1]
else:
    # Default name if you just run "python modeltrain.py"
    EXPERIMENT_NAME = "my_experiment"

print(f"starting experiment: {EXPERIMENT_NAME}")
try:
    from cnn import create_baseline_model
except ImportError:
    print("\n File Error ")
    exit()

# dataset config
DATA_DIR = Path("plant_disease_dataset") 
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32 
EPOCHS_TO_TRAIN = 10 
RANDOM_SEED = 123
USE_SUBSET = True
SUBSET_FRACTION = 0.1 
print(f"Loading image data from: {DATA_DIR}")

# verify classes 
import os
class_names = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
if "__pycache__" in class_names: class_names.remove("__pycache__")
print(f"\nFound the following classes: {class_names}")
NUM_CLASSES = len(class_names)
print(f"Total number of classes: {NUM_CLASSES}")

#data loading
#using a fraction of the data
if USE_SUBSET:
    print(f"\nUtizlizing subset mode (Using {SUBSET_FRACTION*100}% of data) ---")
    subset_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        label_mode='int',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED,
        validation_split=(1.0 - SUBSET_FRACTION), 
        subset="training" 
    )
    dataset_size_batches = tf.data.experimental.cardinality(subset_ds).numpy()
    if dataset_size_batches == 0:
        print(f"\nerror ther were 0 loaded images. check the DATA_DIR path.")
        exit()

    val_size = int(dataset_size_batches * 0.2) 
    train_size = dataset_size_batches - val_size
    
    train_dataset = subset_ds.take(train_size)
    validation_dataset = subset_ds.skip(train_size)
    
    print(f"Splitting subset into: {train_size} train / {val_size} val batches")
    test_dataset = validation_dataset 

else:
    #full mode, loads the full dataset (70/15/15 split)
    print("\nUtilizing full mode")
    full_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, label_mode='int', image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, shuffle=True, seed=RANDOM_SEED
    )
    dataset_size_batches = tf.data.experimental.cardinality(full_dataset).numpy()
    train_size = int(dataset_size_batches * 0.7)
    val_size = int(dataset_size_batches * 0.15)
    test_size = dataset_size_batches - train_size - val_size
    
    print(f"Splitting into: {train_size} train / {val_size} val / {test_size} test batches")

    train_dataset = full_dataset.take(train_size)
    validation_dataset = full_dataset.skip(train_size).take(val_size)
    test_dataset = full_dataset.skip(train_size + val_size).take(test_size)


#train and build model
print("\nBuilding model...")
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3) 
model = create_baseline_model(input_shape, NUM_CLASSES)
model.summary()
print("\nStarting model training...")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS_TO_TRAIN
)
print("\nTraining complete.")

#eval model on the test set 
print("\nEvaluating Model on the Test Set")
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

#saving model and history
model_filename = f"model_{EXPERIMENT_NAME}.keras"
model.save(model_filename)
print(f"\nModel saved to '{model_filename}'")
history_filename = f"history_{EXPERIMENT_NAME}.json"
with open(history_filename, 'w') as f:
    json.dump(history.history, f)
print(f"Training history saved to '{history_filename}'")
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
print(f"Class names saved to 'class_names.json'")

