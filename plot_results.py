import json
import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) > 1:
    EXPERIMENT_NAME = sys.argv[1]
else:
    print("\nerror")
    print("You must provide an experiment name.")
    print("Ex: python plot_results.py baseline")
    exit()

HISTORY_FILE = f"history_{EXPERIMENT_NAME}.json"
print(f"Plotting experiement: {EXPERIMENT_NAME} ---")
print(f"Loading training history from '{HISTORY_FILE}'...")

try:
    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)
except FileNotFoundError:
    print("\nerror")
    print(f"Could not find '{HISTORY_FILE}'.")
    exit()
print("History loaded successfully.")

#Get data fromhistory file 
acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']
epochs_range = range(len(acc))

#Plot Accuracy
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title(f'{EXPERIMENT_NAME} - Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
acc_filename = f"plots_accuracy_{EXPERIMENT_NAME}.png"
plt.savefig(acc_filename)
print(f"Saved '{acc_filename}'")

#Plot Loss 
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title(f'{EXPERIMENT_NAME} - Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
loss_filename = f"plots_loss_{EXPERIMENT_NAME}.png"
plt.savefig(loss_filename)
print(f"Saved '{loss_filename}'")

print("\nPlot generation complete.")

