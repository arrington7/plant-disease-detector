import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#Model Config
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CLASSES = 15

def create_baseline_model(input_shape, num_classes):
    """baseline CNN model for image classification.
        Arguements:
        input_shape (tuple): The shape of the input images (e.g., (256, 256, 3)).
        num_classes (int): The number of output classes."""
    model = Sequential([
        #Loading and Preprocessing
        layers.Input(shape=input_shape),
        layers.Rescaling(1./255), #scales values from [0, 255] down to [0, 1].
        
        #Data Augmentation 
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),      

        #Feature Extraction
        #uses 32 filters (3x3 in size) to find simple patterns like edges or lines.
        #'padding="same"' makes it so the output image size is the same as the input.
        #max Pooling downsamples the image, keeping only the most important features. 
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        #64 filters, combines the simple patterns from the block above.
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        #128 filters
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        #Classification 
        layers.Flatten(),
        #decision layer with 128 neurons. looks at the features from flatten layer
        #and learns which combinations lead to what disease.
        layers.Dense(128, activation='relu'),
        #dropout layer, (used for overfitting)
        #randomly turns off 50% of the neurons during training, forces the model to learn redundant patterns.
        layers.Dropout(0.5),
        #softmax converts the outputs into probabilities for each class.
        layers.Dense(num_classes, activation='softmax')
    ])
    
    #Compile 
    model.compile(
        optimizer='adam', #Adaptive Moment Estimation
        #loss function.
        loss='sparse_categorical_crossentropy',
        #accuracy 
        metrics=['accuracy']
    )
    return model