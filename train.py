import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large

img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values from [0, 255] to [0, 1]
    rotation_range=20, # Randomly rotate images up to 20 degrees
    zoom_range=0.2, # Randomly zoom in/out by up to 20%
    horizontal_flip=True, # Randomly flip images horizontally
    validation_split=0.2   # 20% of training data used as validation
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    "data/train/",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training"
)

validation_generator = val_datagen.flow_from_directory(
    "data/train/",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation"
)


print("Training samples:", train_generator.samples) #Found 492 images belonging to 2 classes.
print("Validation samples:", validation_generator.samples) #Found 122 images belonging to 2 classes.


base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False   # freeze pretrained layers


# Input layer
inputs = tf.keras.Input(shape=(224, 224, 3))

# Pass through base model
x = base_model(inputs, training=False)

# Global average pooling to flatten features
x = layers.GlobalAveragePooling2D()(x)

# Optional dropout for regularization
x = layers.Dropout(0.2)(x)

# Final binary classification layer
outputs = layers.Dense(1, activation='sigmoid')(x)

# Define the model
model = tf.keras.Model(inputs, outputs)

def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)  # convert probabilities to 0 or 1
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float'))
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float'))
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float'))
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    
    return 2 * (precision * recall) / (precision + recall + 1e-7)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', f1_score]
)

model.summary()

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)


# Fine Tune Model
# Unfreeze base model
base_model.trainable = True

# Lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', f1_score]
)

# Fine-tune
history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5
)

model.save("mobilenetv3_santa.h5")
print("Model saved as mobilenetv3_santa.h5")
