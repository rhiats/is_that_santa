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


print("Training samples:", train_generator.samples)
print("Validation samples:", validation_generator.samples)


"""
base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False   # freeze pretrained layers

"""