<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.21+-orange?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Poetry-Dependency%20Manager-cyan?style=for-the-badge&logo=poetry&logoColor=white" alt="Poetry">
</p>

# 🤟 Sign Language Classifier

> **Clasificador de Lenguaje de Señas con Redes Neuronales Convolucionales**

Un proyecto de deep learning para reconocimiento de señas de mano utilizando una arquitectura CNN moderna con TensorFlow/Keras.

---

## 📋 Sobre el Proyecto

Este proyecto comenzó como una implementación básica de clasificación de imágenes y está siendo **modernizado** para cumplir con las mejores prácticas actuales de desarrollo en Python y Machine Learning.

### 🔄 Estado de Modernización

| Aspecto                 | Antes                   | Ahora                              |
| ----------------------- | ----------------------- | ---------------------------------- |
| Gestión de dependencias | `requirements.txt`      | Poetry                             |
| APIs de TensorFlow      | Deprecadas              | Modernas (tf.keras)                |
| Arquitectura CNN        | Básica                  | BatchNorm + Dropout + Augmentation |
| Estructura del código   | Monolítico              | Modular                            |
| Testing                 | Ninguno                 | Property-Based Testing             |
| Seguridad               | Carga de modelo insegura | `safe_mode=True` + límites de tamaño |

---

## 🧠 Arquitectura del Modelo

```
Input (150x150x3)
       │
       ▼
┌─────────────────────┐
│  Conv2D (32) + BN   │──► ReLU ──► MaxPool
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Conv2D (64) + BN   │──► ReLU ──► MaxPool
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Conv2D (128) + BN  │──► ReLU ──► MaxPool
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Dense (256)        │──► ReLU ──► Dropout(0.5)
│  Dense (128)        │──► ReLU ──► Dropout(0.3)
│  Dense (N clases)   │──► Softmax
└─────────────────────┘
```

**Características clave:**

- 🔹 **BatchNormalization** después de cada capa convolucional para estabilizar el entrenamiento
- 🔹 **Dropout** en capas densas para prevenir overfitting
- 🔹 **Filtros progresivos** (32 → 64 → 128) para capturar características de diferentes niveles

### Transfer Learning (opcional)

Si tienes más datos, puedes activar transfer learning con MobileNetV3Small:

```bash
SIGN_CLASSIFIER_USE_TRANSFER_LEARNING=true poetry run sign-classifier train
```

Esto congela el backbone preentrenado en ImageNet y solo entrena el clasificador final.

---

## 🚀 Instalación

### Prerrequisitos

- Python 3.12 o superior
- [Poetry](https://python-poetry.org/docs/#installation) (gestor de dependencias)

### Pasos

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/sign-classifier.git
cd sign-classifier

# Instalar dependencias con Poetry
poetry install

# Activar el entorno virtual
poetry shell
```

### Verificar instalación

```bash
# Ejecutar tests para verificar que todo funciona
poetry run pytest -v
```

---

## 📁 Estructura del Proyecto

```
sign-classifier/
├── src/
│   └── sign_classifier/        # Paquete principal (código modernizado)
│       ├── __init__.py
│       ├── batch_predict.py    # Predicción por lotes
│       ├── cli.py              # Interfaz de línea de comandos
│       ├── config.py           # Configuración centralizada
│       ├── data_loader.py      # Carga y augmentación de datos
│       ├── evaluate.py         # Evaluación del modelo
│       ├── exceptions.py       # Excepciones personalizadas
│       ├── export.py           # Exportación a SavedModel/TFLite
│       ├── generate_augmented_data.py    # Generación de datos aumentados
│       ├── integrate_external_dataset.py # Integración de datasets externos
│       ├── model.py            # Arquitectura CNN mejorada
│       ├── predict.py          # Módulo de predicción
│       └── train.py            # Script de entrenamiento moderno
├── tests/                      # Tests unitarios y de integración
├── data/                       # Datos de entrenamiento y validación
├── modelo/                     # Modelos entrenados (.keras)
├── metrics/                    # Métricas de entrenamiento (CSV/JSON)
├── Dockerfile                  # Imagen Docker del proyecto
├── pyproject.toml              # Configuración de Poetry
├── poetry.lock                 # Lock de dependencias
├── LICENSE                     # Licencia MIT
└── README.md
```

---

## 💻 Uso

El proyecto expone una interfaz de línea de comandos (CLI) a través del comando `sign-classifier`.

```bash
# Ver ayuda
poetry run sign-classifier --help

# Entrenar
poetry run sign-classifier train

# Evaluar
poetry run sign-classifier evaluate

# Predecir una imagen
poetry run sign-classifier predict ruta/a/imagen.jpg

# Predecir un directorio completo
poetry run sign-classifier batch-predict ./carpeta_imagenes --output predicciones.csv

# Exportar modelo
poetry run sign-classifier export
```

### Entrenamiento

```bash
# Entrenar el modelo usando el módulo modernizado
poetry run python -m sign_classifier.train

# El mejor modelo se guardará en ./modelo/modelo.keras
```

### Evaluación

```bash
# Evaluar el modelo entrenado con el dataset de validación
poetry run python -m sign_classifier.evaluate
```

### Predicción

```python
# Ejemplo de predicción segura con el módulo modernizado
from sign_classifier.predict import (
    load_and_preprocess_image,
    load_model_safe,
    predict_class,
)
from sign_classifier.config import CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH, MODEL_PATH

# Cargar modelo entrenado en modo seguro
model = load_model_safe(MODEL_PATH)

# Preprocesar imagen
image = load_and_preprocess_image("ruta/a/imagen.jpg", (IMAGE_HEIGHT, IMAGE_WIDTH))

# Predecir clase
clase = predict_class(model, image, CLASSES)
print(f"Clase predicha: {clase}")
```

### Uso programático del modelo

```python
from sign_classifier.model import create_model
from sign_classifier.config import IMAGE_SHAPE, NUM_CLASSES

# Crear modelo desde cero
model = create_model(IMAGE_SHAPE, NUM_CLASSES)

# O cargar modelo entrenado de forma segura
from sign_classifier.predict import load_model_safe
model = load_model_safe("modelo/modelo.keras")
```

### Configuración por variables de entorno

Toda la configuración puede sobrescribirse con variables de entorno usando el
prefijo `SIGN_CLASSIFIER_`:

```bash
SIGN_CLASSIFIER_EPOCHS=50 \
SIGN_CLASSIFIER_BATCH_SIZE=16 \
SIGN_CLASSIFIER_LEARNING_RATE=0.001 \
poetry run sign-classifier train
```

Variables disponibles:

| Variable | Descripción | Default |
|---|---|---|
| `SIGN_CLASSIFIER_EPOCHS` | Épocas de entrenamiento | 20 |
| `SIGN_CLASSIFIER_BATCH_SIZE` | Tamaño del batch | 32 |
| `SIGN_CLASSIFIER_LEARNING_RATE` | Learning rate de Adam | 0.0004 |
| `SIGN_CLASSIFIER_TRAINING_DATA_PATH` | Ruta de entrenamiento | `./data/entrenamiento` |
| `SIGN_CLASSIFIER_VALIDATION_DATA_PATH` | Ruta de validación | `./data/validacion` |
| `SIGN_CLASSIFIER_MODEL_PATH` | Ruta del modelo | `./modelo/modelo.keras` |
| `SIGN_CLASSIFIER_CLASSES` | Lista de clases separadas por coma | `a,b,c` |
| `SIGN_CLASSIFIER_USE_TRANSFER_LEARNING` | Usar transfer learning | `false` |
| `SIGN_CLASSIFIER_TRANSFER_LEARNING_BACKBONE` | Backbone preentrenado | `mobilenet_v3` |

### Métricas de entrenamiento

Al entrenar se generan automáticamente:

- `metrics/history.csv`: métricas por época.
- `metrics/metrics.json`: resumen con mejor val_accuracy, final loss, etc.

### Predicción por lotes

```bash
poetry run sign-classifier batch-predict ./imagenes --output predicciones.csv
```

El CSV incluye la clase predicha, confianza y probabilidad por clase.

### Exportar modelo

```bash
# SavedModel + TFLite
poetry run sign-classifier export

# Solo TFLite
poetry run sign-classifier export --format tflite --tflite-path modelo.tflite
```

### Docker

```bash
# Construir imagen
docker build -t sign-classifier .

# Entrenar
docker run -v $(pwd)/data:/app/data -v $(pwd)/modelo:/app/modelo sign-classifier train

# Predecir
docker run -v $(pwd)/modelo:/app/modelo -v $(pwd)/imagen.jpg:/app/imagen.jpg sign-classifier predict /app/imagen.jpg
```

### Generar más datos de entrenamiento

Si tienes pocas imágenes, puedes generar variantes aumentadas a partir de las
imágenes reales existentes. Esto mantiene la coherencia porque parte de manos
reales haciendo señas:

```bash
# Generar 20 variantes por imagen de entrenamiento
poetry run python -m sign_classifier.generate_augmented_data \
    --input-dir ./data/entrenamiento \
    --output-dir ./data/entrenamiento \
    --variants-per-image 20

# Generar 5 variantes por imagen de validación
poetry run python -m sign_classifier.generate_augmented_data \
    --input-dir ./data/validacion \
    --output-dir ./data/validacion \
    --variants-per-image 5
```

Las transformaciones aplicadas son realistas: rotación, zoom, traslación,
brillo, contraste, saturación, desenfoque, ruido leve, gamma y recortes.

### Integrar datasets externos

Si descargas un dataset público de ASL (por ejemplo, de Kaggle o Roboflow),
puedes integrarlo fácilmente:

```bash
poetry run python -m sign_classifier.integrate_external_dataset \
    --source-dir ./dataset_externo \
    --target-dir ./data/entrenamiento \
    --split-ratio 0.8 \
    --val-target-dir ./data/validacion
```

---

## 🧪 Testing

El proyecto utiliza **Property-Based Testing** con Hypothesis para validar propiedades universales del modelo:

```bash
# Ejecutar todos los tests
poetry run pytest

# Ejecutar con verbose
poetry run pytest -v

# Ejecutar tests específicos
poetry run pytest tests/test_model.py -v
```

---

## 🏭 Despliegue en Producción

### Opción 1: TensorFlow Serving

```bash
# Exportar modelo para serving
model.save("modelo_serving/1/")

# Ejecutar TensorFlow Serving con Docker
docker run -p 8501:8501 \
  --mount type=bind,source=$(pwd)/modelo_serving,target=/models/sign_classifier \
  -e MODEL_NAME=sign_classifier \
  tensorflow/serving
```

### Opción 2: API REST con FastAPI

```python
from fastapi import FastAPI, UploadFile
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model("modelo/modelo.keras")

@app.post("/predict")
async def predict(file: UploadFile):
    # Preprocesar imagen y predecir
    ...
```

### Opción 3: AWS Lambda + API Gateway

1. Empaquetar modelo con dependencias
2. Crear función Lambda
3. Configurar API Gateway como trigger

---

## 📚 Recursos de Aprendizaje

### Libros Recomendados (Packt Publishing)

| Libro                                                                                                                                               | Descripción                                                                        |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **[Deep Learning with TensorFlow and Keras](https://www.packtpub.com/product/deep-learning-with-tensorflow-and-keras-third-edition/9781803232911)** | Guía completa de deep learning con las últimas APIs de TensorFlow 2.x              |
| **[Hands-On Computer Vision with TensorFlow 2](https://www.packtpub.com/product/hands-on-computer-vision-with-tensorflow-2/9781788830645)**         | Proyectos prácticos de visión por computadora incluyendo clasificación de imágenes |
| **[TensorFlow 2.0 Quick Start Guide](https://www.packtpub.com/product/tensorflow-20-quick-start-guide/9781789530759)**                              | Introducción rápida a TensorFlow 2.0 y Keras                                       |
| **[Python Machine Learning](https://www.packtpub.com/product/python-machine-learning-third-edition/9781789955750)**                                 | Fundamentos de ML con scikit-learn y deep learning                                 |
| **[Practical Convolutional Neural Networks](https://www.packtpub.com/product/practical-convolutional-neural-networks/9781788392303)**               | Implementación práctica de CNNs para diferentes aplicaciones                       |

### Documentación Oficial

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)
- [Poetry Documentation](https://python-poetry.org/docs/)

---

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

---

<p align="center">
  <sub>Hecho con ❤️ para la comunidad de desarrolladores</sub>
</p>
