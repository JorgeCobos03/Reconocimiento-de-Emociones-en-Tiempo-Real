# Sistema-de-Detección-de-Emociones-en-Tiempo-Real

Este proyecto implementa un sistema para la detección de emociones faciales utilizando **Mediapipe**, **KNN (K-Nearest Neighbors)** y una interfaz gráfica construida con **PyQt5**. La aplicación analiza imágenes de rostros en tiempo real, extrae características faciales relevantes y clasifica las emociones dominantes. Además, proporciona visualizaciones de los resultados en gráficos de barras y radar.

---

## Características del Proyecto

- **Extracción de características faciales**:
  - Distancia entre los ojos
  - Ancho de la boca
  - Distancia entre cejas
  - Apertura de ojos (izquierdo y derecho)
  
- **Clasificación de emociones**:
  - Feliz (`happy`)
  - Triste (`sad`)
  - Neutral (`neutral`)
  - Enojado (`angry`)

- **Entrenamiento con KNN**:
  - Uso del modelo K-Nearest Neighbors para clasificar las emociones.
  - El modelo entrenado se guarda en un archivo `.pkl` para reutilización.

- **Interfaz de usuario**:
  - Pantalla de bienvenida con instrucciones y una imagen.
  - Detección en tiempo real con cámara integrada.
  - Resultados visuales en gráficos radar y de barras.

---

## Requisitos

- **Python 3.8 o superior**
-**Camara Web HD (720p+)**
- **Bibliotecas necesarias**:
  - `mediapipe`
  - `opencv-python`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `joblib`
  - `pandas`
  - `pyqt5`

---

## Instalación

El siguiente comando pip es para instalar todas las dependencias necesarias para el proyecto. Este comando se asegura de incluir todas las bibliotecas utilizadas.
   ```bash
    pip install opencv-python mediapipe numpy pandas scikit-learn joblib matplotlib pyqt5
   ``` 
## Ejecución

1. El archivo `FaceLandmarksVisualizer.py` se encarga de procesar una imagen de rostro y resaltar puntos faciales específicos utilizando la biblioteca Mediapipe para la detección de mallas faciales y OpenCV para el manejo de imágenes. 

2. El archivo `FeaturesCSV.py` procesa imágenes de rostros para extraer características faciales específicas utilizando Mediapipe y guarda estas características en un archivo CSV, categorizándolas por etiquetas basadas en emociones o categorías presentes en las subcarpetas del conjunto de datos.

3. El archivo `KNN.py` entrenará un modelo KNN con los parámetros dados, medirá su rendimiento en el conjunto de prueba y generará un informe de clasificación. El modelo final será guardado para su uso posterior.

4. Finalmente elarchivo `DeteccionTR` implementa el sistema de detección de emociones basado en puntos faciales y un modelo KNN previamente entrenado, todo integrado en una interfaz gráfica usando PyQt5.
