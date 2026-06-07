# SEGMENTACION DE CELULAS

Comparando 2 tipos de arquitecturas para la segmentacion de celulas:
1. ** Arquitectura U-Net**
2. **Arquitectura Mamba**





Primero se realizo una investigacion sobre el dataset, obtiendo 670 imgenes con su respectiva mascara
para comenzar, primero se construyo un dataset fusionado en la carpeta dataset_fusionado/.

El archivo utils/dataset.py se encarga de cargar el dataset y combinar las mascaras en una sola
que posteriormente se ocupara para el entrenamiento de los modelos.



el dataset https://bbbc.broadinstitute.org/BBBC038

## Entrenamiento U-Net

El entrenamiento usa `src/train_unet.py` sobre el dataset fusionado en `dataset_fusionado/images` y `dataset_fusionado/masks`. El script valida los pares imagen/mascara, entrena con `tf.data` y guarda el modelo junto con las metricas en el directorio indicado con `--output-dir`.

### Preparar entorno

Activar la `.venv` del repo:

```bash
source .venv/bin/activate
```

Instalar dependencias minimas si el entorno no las tiene:

```bash
python -m pip install tensorflow pillow matplotlib
```

En este entorno ya estaba disponible TensorFlow 2.21.0 con Keras 3.14.1, NumPy 2.4.6, Pillow 12.2.0 y Matplotlib 3.10.9.

### Verificar dataset sin entrenar

```bash
python src/train_unet.py --dry-run
```

Resultado obtenido: 670 pares imagen/mascara detectados y estimacion de 0.65 GiB si se cargaran como arreglos NumPy.

### Smoke train

Corrida minima para verificar que TensorFlow, el modelo, el dataset y el guardado funcionan:

```bash
python src/train_unet.py --device cpu --epochs 1 --batch-size 1 --filters 4 --steps-per-epoch 1 --validation-steps 1 --output-dir outputs/unet-smoke
```

Resultado obtenido: 3.90 segundos, `accuracy=0.8617`, `loss=0.6637`, `val_accuracy=0.8761`, `val_loss=0.6867`. Artefactos guardados en `outputs/unet-smoke/`.

### Entrenamiento acotado usado en este entorno

Como TensorFlow no pudo usar CUDA/GPU en esta maquina, se ejecuto una corrida CPU acotada para evitar saturar el equipo:

```bash
python src/train_unet.py --device cpu --epochs 5 --batch-size 2 --filters 8 --steps-per-epoch 20 --validation-steps 5 --output-dir outputs/unet-cpu-safe
```

Resultado obtenido: 10.10 segundos, 670 pares totales, 536 de entrenamiento y 134 de validacion. Metricas finales: `accuracy=0.9192`, `loss=0.2181`, `val_accuracy=0.5420`, `val_loss=0.9965`. Artefactos guardados en `outputs/unet-cpu-safe/`:

```text
outputs/unet-cpu-safe/model.keras
outputs/unet-cpu-safe/history.json
outputs/unet-cpu-safe/metrics.json
```

### CUDA/GPU en este entorno

TensorFlow 2.21.0 esta compilado con soporte CUDA, pero no pudo registrar GPU por librerias CUDA/cuDNN no disponibles o no cargables. La verificacion mostro `physical_gpus=[]` y el entrenamiento se hizo por CPU. Para intentar GPU en otra maquina, instalar las librerias CUDA/cuDNN compatibles con TensorFlow y ejecutar con `--device auto` o `--device gpu`.

### Entrenamiento mas completo

Cuando haya GPU funcional o se acepte una corrida CPU mas larga, usar mas epocas, mas filtros y quitar los limites de pasos:

```bash
python src/train_unet.py --device auto --epochs 20 --batch-size 4 --filters 32 --output-dir outputs/unet-full
```
