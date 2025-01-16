import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Inicializar Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Función para extraer puntos faciales por región
def extract_face_regions(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Escala de grises
    image = cv2.resize(image, (255, 255))  # Asegurar tamaño 48x48
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Mediapipe necesita RGB aunque venga de escala de grises

    results = face_mesh.process(image_rgb)
    features = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w = image.shape[:2]

            # Extraer coordenadas de puntos
            landmarks = np.array([(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark])

            # Calcular distancias y características
            eye_dist = np.linalg.norm(landmarks[468] - landmarks[473])  # Distancia entre ojos
            mouth_width = np.linalg.norm(landmarks[78] - landmarks[308])  # Ancho de boca
            eyebrow_dist = np.linalg.norm(landmarks[65] - landmarks[295])  # Distancia entre cejas
            eye_openness_left = np.linalg.norm(landmarks[159] - landmarks[145])
            eye_openness_right = np.linalg.norm(landmarks[386] - landmarks[374])

            features = [eye_dist, mouth_width, eyebrow_dist, eye_openness_left, eye_openness_right]

    return features

# Guardar en CSV
def process_dataset(data_dir, output_csv='face_features.csv'):
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['label', 'eye_distance', 'mouth_width', 'eyebrow_distance', 'eye_openness_left', 'eye_openness_right'])

        # Recorrer subcarpetas (angry, happy, etc.)
        for subdir, _, files in os.walk(data_dir):
            label = os.path.basename(subdir)  # La carpeta es la etiqueta

            for filename in files:
                image_path = os.path.join(subdir, filename)
                features = extract_face_regions(image_path)
                
                if features:
                    writer.writerow([label] + features)

# Procesar el conjunto de datos FER2013
data_dir = 'C:/Users/cobos/Documents/Vision Artificial/Proyecto/KNN_dlbi/Data'  # Ruta donde están las subcarpetas (angry, happy, etc.)
process_dataset(data_dir)