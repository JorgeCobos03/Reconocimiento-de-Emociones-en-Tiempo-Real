import cv2
import mediapipe as mp

# Inicializar Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Cargar imagen
image_path ='C:/Users/cobos/Documents/Vision Artificial/Proyecto/KNN_dlbi/Data/happy/PrivateTest_218533.jpg' 
image = cv2.imread(image_path)
image = cv2.resize(image,(255,255))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Procesar la imagen con Mediapipe
results = face_mesh.process(image_rgb)

# Función para dibujar puntos específicos
def draw_landmark(image, landmark, color):
    x = int(landmark.x * image.shape[1])
    y = int(landmark.y * image.shape[0])
    cv2.circle(image, (x, y), 3, color, -1)

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Dibujar los puntos específicos
        draw_landmark(image, face_landmarks.landmark[468], (0, 255, 0))  # Punto de ojo
        draw_landmark(image, face_landmarks.landmark[473], (0, 255, 0))  # Punto de ojo
        draw_landmark(image, face_landmarks.landmark[78], (255, 0, 0))   # Punto de boca
        draw_landmark(image, face_landmarks.landmark[308], (255, 0, 0))  # Punto de boca
        draw_landmark(image, face_landmarks.landmark[65], (0, 0, 255))   # Punto de ceja
        draw_landmark(image, face_landmarks.landmark[295], (0, 0, 255))  # Punto de ceja
        draw_landmark(image, face_landmarks.landmark[159], (255, 255, 0))# Punto de ojo izquierdo
        draw_landmark(image, face_landmarks.landmark[145], (255, 255, 0))# Punto de ojo izquierdo
        draw_landmark(image, face_landmarks.landmark[386], (255, 0, 255))# Punto de ojo derecho
        draw_landmark(image, face_landmarks.landmark[374], (255, 0, 255))# Punto de ojo derecho

cv2.imshow('Face Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
