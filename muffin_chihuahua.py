import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#%%
# Configuración
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
#%%
# Generadores de datos
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
#%%
train_generator = train_datagen.flow_from_directory(
    'dataset/train/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
#%%
test_generator = test_datagen.flow_from_directory(
    'dataset/test/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
#%%
# Modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Clasificación binaria
])
#%%
# Compilación
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#%%
# Entrenamiento
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator
)
#%%
# Guardar el modelo
model.save('muffin_vs_chihuahua_model.h5')
#%%
# Evaluación del modelo
import numpy as np
from tensorflow.keras.preprocessing import image
#%%
# Cargar el modelo
model = tf.keras.models.load_model('muffin_vs_chihuahua_model.h5')
#%%
# Cargar una imagen de prueba
img_path = 'dataset/test/chihuahua/img_4_226.jpg'  # Cambia a tu imagen
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
#%%
# Predicción
prediction = model.predict(img_array)
print("Chihuahua" if prediction[0][0] < 0.5 else "Muffin")
#%%
import matplotlib.pyplot as plt

# Curvas de entrenamiento y validación
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del Modelo')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.legend()
plt.show()