# Custom Pylint rules for the file
# pylint: disable=E0401 W0719 R0913 C0301 W0718 E1101
# E0401:import-error
# W0719:broad-exception-raised
# R0913:too-many-arguments
# C0301:line-too-long
# W0718:broad-exception-caught
# E1101:no-member

from tensorflow import keras
from tensorflow.keras import layers

def build_model_1(num_classes,image_size,num_channels=3):
    """
    Simpler Model.
    """
    model = keras.Sequential([

        # Convolutional layers
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", input_shape=(image_size, image_size, num_channels)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
    
        # Flatten the output from convolutional layers (channels are condensed)
        layers.Flatten(),

        # Dense layers
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),

        # Output layer with softmax activation for multi-class classification
        layers.Dense(num_classes, activation="softmax")
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def build_model_2(num_classes,image_size,num_channels=3):
    """
    Deeper Model.
    """
    model = keras.Sequential([

        # Convolutional layers
        layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(image_size, image_size, num_channels)),
        layers.Conv2D(180,kernel_size=(3,3),activation='relu'),
        layers.MaxPool2D(pool_size=(5, 5)),
        layers.Conv2D(180,kernel_size=(3,3),activation='relu'),
        layers.Conv2D(140,kernel_size=(3,3),activation='relu'),
        layers.Conv2D(100,kernel_size=(3,3),activation='relu'),
        layers.Conv2D(50,kernel_size=(3,3),activation='relu'),
        layers.MaxPool2D(pool_size=(5, 5)),
        # Flatten the output from convolutional layers (channels are condensed)
        layers.Flatten(),

        # Dense layers + dropout
        layers.Dense(180,activation='relu'),
        layers.Dense(120,activation='relu'),
        layers.Dropout(rate=0.25),
        layers.Dense(60,activation='relu'),
        layers.Dropout(rate=0.05),

        # Output layer with softmax activation for multi-class classification
        layers.Dense(num_classes,activation='softmax')
    ])

    model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])

    return model
