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

---

## 🚀 Instalación

### Prerrequisitos

- Python 3.9 o superior
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
│       ├── cli.py              # Interfaz de línea de comandos
│       ├── config.py           # Configuración centralizada
│       ├── data_loader.py      # Carga y augmentación de datos
│       ├── evaluate.py         # Evaluación del modelo
│       ├── exceptions.py       # Excepciones personalizadas
│       ├── model.py            # Arquitectura CNN mejorada
│       ├── predict.py          # Módulo de predicción
│       └── train.py            # Script de entrenamiento moderno
├── tests/
│   ├── __init__.py
│   ├── test_model.py           # Tests de propiedades del modelo
│   └── test_predict.py         # Tests de predicción
├── data/
│   ├── entrenamiento/          # Datos de entrenamiento (por clase)
│   │   ├── a/
│   │   ├── b/
│   │   └── c/
│   └── validacion/             # Datos de validación (por clase)
│       ├── a/
│       ├── b/
│       └── c/
├── modelo/                     # Modelos entrenados (.keras)
├── pyproject.toml              # Configuración de Poetry
├── poetry.lock                 # Lock de dependencias
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
