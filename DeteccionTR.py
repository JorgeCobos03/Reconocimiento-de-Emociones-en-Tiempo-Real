import sys
import cv2
import mediapipe as mp
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton, QStackedWidget, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt

# Cargar el modelo
modelo_cargado = joblib.load('C:/Users/cobos/Documents/Vision Artificial/Proyecto/KNN_dlbi/modelo_knn.pkl')

# Inicializar Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

class WelcomePage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Etiqueta de bienvenida
        welcome_label = QLabel("Bienvenido al Sistema de Detección de Emociones")
        welcome_label.setFont(QFont("Times New Roman", 24, QFont.Bold))
        welcome_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(welcome_label)

        # Cargar imagen
        image_label = QLabel()
        pixmap = QPixmap("emotions.jpg")  
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(image_label)

        # Botón de comenzar detección
        start_button = QPushButton("Comenzar Detección")
        start_button.setFont(QFont("Times New Roman", 16))
        start_button.setStyleSheet("background-color: #0078D7; color: white; padding: 10px; border-radius: 8px;")
        start_button.clicked.connect(self.start_detection)
        layout.addWidget(start_button)

        layout.setSpacing(20)
        layout.setContentsMargins(50, 50, 50, 50)

    def start_detection(self):
        self.parent.switch_to_detection()

class DetectionPage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Temporizador y otros elementos de UI
        self.timer_label = QLabel("Tiempo restante: 20s")
        self.timer_label.setFont(QFont("Times New Roman", 18))
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("color: white; background-color: #0078D7; padding: 5px; border-radius: 8px;")
        self.layout.addWidget(self.timer_label)

        # Otros elementos de la UI
        self.camera_layout = QVBoxLayout()
        self.camera_label = QLabel()
        self.camera_label.setStyleSheet("border: 2px solid black; border-radius: 10px;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_layout.addWidget(self.camera_label)
        self.layout.addLayout(self.camera_layout)

        # Botón de mostrar resultados
        self.results_button = QPushButton("Mostrar Resultados")
        self.results_button.setFont(QFont("Times New Roman", 16))
        self.results_button.setStyleSheet("background-color: #28A745; color: white; padding: 10px; border-radius: 8px;")
        self.results_button.setEnabled(False)
        self.results_button.clicked.connect(self.parent.switch_to_results)
        self.layout.addWidget(self.results_button)

        # Botón para repetir la detección
        self.repeat_button = QPushButton("Repetir Detección")
        self.repeat_button.setFont(QFont("Times New Roman", 16))
        self.repeat_button.setStyleSheet("background-color: #FF5733; color: white; padding: 10px; border-radius: 8px;")
        self.repeat_button.setEnabled(False)
        self.repeat_button.clicked.connect(self.reset_detection)
        self.layout.addWidget(self.repeat_button)

        self.results_button.hide()
        self.repeat_button.hide()

        # Configuración de la cámara y temporizadores
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.time_left = 20

        self.detection_timer = QTimer(self)
        self.detection_timer.timeout.connect(self.update_timer)

        self.emotions_count = {
            'happy': 0,
            'sad': 0,
            'neutral': 0,
            'angry': 0
        }

        # Establecer la duración de la detección
        self.detection_duration = self.time_left

        # Establecer márgenes y espaciado
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(50, 50, 50, 50)

    def start_detection(self):
        self.initialize_camera()
        self.timer.start(30)
        self.detection_timer.start(1000)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        features, frame_with_points = self.extract_face_regions(frame)

        if features:
            features = np.array(features).reshape(1, -1)
            prediction = modelo_cargado.predict(features)[0]
            self.emotions_count[prediction] += 1
            cv2.putText(frame_with_points, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame_with_points, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.camera_label.setPixmap(pixmap)

    def update_timer(self):
        self.time_left -= 1
        self.timer_label.setText(f"Tiempo restante: {self.time_left}s")
        if self.time_left == 0:
            self.detection_timer.stop()
            self.timer.stop()
            self.cap.release()
            self.results_button.setEnabled(True)
            self.repeat_button.setEnabled(True)
            self.results_button.show()
            self.repeat_button.show()

    def reset_detection(self):
        # Resetear todos los contadores y temporizadores
        self.time_left = 20
        self.detection_duration = self.time_left
        self.emotions_count = {'happy': 0, 'sad': 0, 'neutral': 0, 'angry': 0}
        self.timer_label.setText("Tiempo restante: 20s")
        self.results_button.hide()
        self.repeat_button.hide()

        # Reiniciar la cámara y la detección
        self.initialize_camera()

        # Iniciar nuevamente la detección
        self.start_detection()

    def initialize_camera(self):
        # Detener la cámara si ya está abierta
        if self.cap.isOpened():
            self.cap.release()
        # Reabrir la cámara
        self.cap = cv2.VideoCapture(0)

    def extract_face_regions(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        features = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = image.shape[:2]
                landmarks = np.array([(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark])
                eye_dist = np.linalg.norm(landmarks[468] - landmarks[473])
                mouth_width = np.linalg.norm(landmarks[78] - landmarks[308])
                eyebrow_dist = np.linalg.norm(landmarks[65] - landmarks[295])
                eye_openness_left = np.linalg.norm(landmarks[159] - landmarks[145])
                eye_openness_right = np.linalg.norm(landmarks[386] - landmarks[374])
                features = [eye_dist, mouth_width, eyebrow_dist, eye_openness_left, eye_openness_right]

                for (x, y) in landmarks:
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        return features, image

class ResultsPage(QWidget):
    def __init__(self, emotions_count, detection_duration):
        super().__init__()
        self.emotions_count = emotions_count
        self.detection_duration = detection_duration 
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Título
        results_label = QLabel("Resultados de la Detección")
        results_label.setFont(QFont("Times New Roman", 22, QFont.Bold))
        results_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(results_label)

        # Crear y mostrar las gráficas
        self.create_radar_chart()
        self.create_bar_chart()

        # Crear layout para gráficas
        graphics_layout = QHBoxLayout()
        layout.addLayout(graphics_layout)

        # Añadir gráficas
        radar_label = QLabel()
        radar_label.setPixmap(QPixmap("radar_chart.png"))
        radar_label.setAlignment(Qt.AlignCenter)
        graphics_layout.addWidget(radar_label)

        bar_label = QLabel()
        bar_label.setPixmap(QPixmap("bar_chart.png"))
        bar_label.setAlignment(Qt.AlignCenter)
        graphics_layout.addWidget(bar_label)

        # Descripción de los resultados
        self.add_results_description()

        # Botón para finalizar
        finalize_button = QPushButton("Finalizar")
        finalize_button.setFont(QFont("Times New Roman", 16))
        finalize_button.setStyleSheet("background-color: #FF0000; color: white; padding: 10px; border-radius: 8px;")
        finalize_button.clicked.connect(sys.exit)
        layout.addWidget(finalize_button)

    def add_results_description(self):
        total_detections = sum(self.emotions_count.values())
        description = ""

        if total_detections > 0:
            for emotion, count in self.emotions_count.items():
                percentage = (count / total_detections) * 100
                description += f"{emotion}: {percentage:.2f}% ({count} detecciones)\n"

            dominant_emotion = max(self.emotions_count, key=self.emotions_count.get)
            description += f"\nEmoción dominante: {dominant_emotion}\n"
        else:
            description = "No se detectaron emociones durante la sesión."

        description_label = QLabel(description)
        description_label.setFont(QFont("Times New Roman", 14))
        description_label.setAlignment(Qt.AlignLeft)
        description_label.setStyleSheet("padding: 10px; background-color: #F0F0F0; border: 1px solid #D3D3D3;")
        description_label.setWordWrap(True)
        self.layout().addWidget(description_label)

    def create_radar_chart(self):
        labels = list(self.emotions_count.keys())
        values = list(self.emotions_count.values())

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        ax.fill(angles, values, color='green', alpha=0.25)
        ax.plot(angles, values, color='green', linewidth=2)

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

        plt.tight_layout()
        plt.savefig('radar_chart.png')

    def create_bar_chart(self):
        emotions = list(self.emotions_count.keys())
        counts = list(self.emotions_count.values())

        emotion_colors = {
            'happy': 'yellow',
            'sad': 'darkblue',
            'neutral': 'gray',
            'angry': 'red'
        }

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(emotions, counts, color=[emotion_colors[emotion] for emotion in emotions])
        ax.set_xlabel('Emoción')
        ax.set_ylabel('Cantidad')
        ax.set_title('Distribución de Emociones Detectadas')

        plt.tight_layout()
        plt.savefig('bar_chart.png')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Detección de Emociones")
        self.setGeometry(100, 100, 800, 600)
        self.stacked_widget = QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)

        # Inicializar páginas
        self.welcome_page = WelcomePage(self)
        self.detection_page = DetectionPage(self)
        self.results_page = None

        self.stacked_widget.addWidget(self.welcome_page)
        self.stacked_widget.addWidget(self.detection_page)

    def switch_to_detection(self):
        self.stacked_widget.setCurrentWidget(self.detection_page)
        self.detection_page.start_detection()

    def switch_to_results(self):
        self.results_page = ResultsPage(self.detection_page.emotions_count, self.detection_page.detection_duration)
        self.stacked_widget.addWidget(self.results_page)
        self.stacked_widget.setCurrentWidget(self.results_page)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
