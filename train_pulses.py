# train_pulses.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# ----------------------------
# 1️⃣ Paths and Parameters
# ----------------------------
train_data_dir = 'datasets/pulses/train_data'  # your pulses train folder
img_height, img_width = 224, 224
batch_size = 32
epochs = 20
num_classes = 3  # Anthracnose, Bacterial_spot, Healthy

# ----------------------------
# 2️⃣ Data Generators
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% for validation
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ----------------------------
# 3️⃣ Model Definition
# ----------------------------
model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ----------------------------
# 4️⃣ Model Training
# ----------------------------
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs
)

# ----------------------------
# 5️⃣ Save Model
# ----------------------------
os.makedirs('models', exist_ok=True)
model.save('models/pulses_disease_model.h5')
print("✅ Model saved as models/pulses_disease_model.h5")
