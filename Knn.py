import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Cargar datos desde CSV
data = pd.read_csv('C:/Users/cobos/Documents/Vision Artificial/Proyecto/KNN_dlbi/face_features.csv')

# Separar características (X) y etiquetas (y)
X = data.drop(columns=['label']) 
y = data['label']  

# Dividir en conjunto de entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar modelo KNN 
knn = KNeighborsClassifier(
    n_neighbors=15,        # Número de vecinos
    weights='distance',    # Ponderación por distancia
    metric='euclidean'     # Métrica de distancia euclidiana
)

# Entrenar el modelo con el conjunto de entrenamiento
knn.fit(X_train, y_train)

# Hacer predicciones con el conjunto de prueba
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Guardar el modelo entrenado en un archivo
joblib.dump(knn, 'modelo_knn.pkl')
