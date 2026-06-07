import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

from models.unet import build_unet

# 1. Cargar las rutas de las imágenes y máscaras
img_dir = Path("dataset_fusionado/images")
mask_dir = Path("dataset_fusionado/masks")

img_paths = sorted(list(img_dir.glob("*.png")))
mask_paths = sorted(list(mask_dir.glob("*.png")))

# 2. Cargar todas las imágenes en memoria y normalizar
X = np.array([np.array(Image.open(p)) / 255.0 for p in img_paths], dtype=np.float32)
# Las máscaras necesitan canal de color (H, W, 1) y normalización
y = np.array([np.expand_dims(np.array(Image.open(p)) / 255.0, axis=-1) for p in mask_paths], dtype=np.float32)

# 3. Separar en Entrenamiento y Validación (80% / 20%)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Instanciar U-Net
unet = build_unet(input_size=(256, 256, 3), n_filters=32, n_classes=1)

# 5. Compilar (Usar BinaryCrossentropy para segmentación binaria)
unet.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# 6. Entrenar
results = unet.fit(
    X_train, y_train,
    batch_size=16,  # 32 puede ser muy pesado para memoria a 256x256
    epochs=20,
    validation_data=(X_valid, y_valid)
)
