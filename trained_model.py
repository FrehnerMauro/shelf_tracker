import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Ordner, der die Trainingsbilder enthält
DATASET_DIR = "/Users/maurofrehner/Library/CloudStorage/OneDrive-Persönlich/Mauro/Schule/FHGraubünden/Programme/Python/Landwasserroboter/Bildbearbeitung2/Projekt/dataset"  # Erstelle einen Ordner "dataset" und füge Bilderklassen als Unterordner hinzu
IMG_SIZE = 128  # Bildgröße, auf die alle Bilder skaliert werden

def load_images_from_folder(folder_path):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder_path))  # Sortiere die Klassen, um konsistente Zuordnungen sicherzustellen
    class_dict = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    try:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        images.append(img)
                        labels.append(class_dict[class_name])
                    except Exception as e:
                        print(f"Fehler beim Verarbeiten des Bildes {img_path}: {e}")
    return np.array(images), np.array(labels), class_dict

def preprocess_data(images, labels):
    images = images / 255.0  # Normalisierung der Bilddaten
    num_classes = len(np.unique(labels))  # Stelle sicher, dass die Anzahl der Klassen korrekt ist
    if np.max(labels) >= num_classes:
        num_classes = np.max(labels) + 1  # Passe die Anzahl der Klassen an, um sicherzustellen, dass alle Labels abgedeckt sind
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    return images, labels

# Lade die Bilder und Labels
images, labels, class_dict = load_images_from_folder(DATASET_DIR)

# Sicherstellen, dass es mindestens eine Klasse gibt
if len(class_dict) == 0:
    raise ValueError("Keine Klassen im angegebenen Dataset-Verzeichnis gefunden.")

images, labels = preprocess_data(images, labels)

# Überprüfe, ob die Labels korrekt sind
num_classes = len(class_dict)
if labels.shape[1] != num_classes:
    raise ValueError(f"Die Anzahl der Klassen in den Labels ({labels.shape[1]}) stimmt nicht mit der Anzahl der gefundenen Klassen ({num_classes}) überein.")

# Aufteilen der Daten in Trainings- und Validierungsdaten
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Erstelle das CNN-Modell
model = models.Sequential([
    layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_dict), activation='softmax')
])

# Kompilieren des Modells
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trainieren des Modells
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Speichern des trainierten Modells
model.save("trained_model.h5")

# Video-Stream von der Webcam vorhersagen
def predict_from_video_stream(model, class_dict):
    cap = cv2.VideoCapture(0)  # Webcam-Stream starten
    if not cap.isOpened():
        raise IOError("Kamera konnte nicht geöffnet werden.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)
        class_name = [name for name, idx in class_dict.items() if idx == predicted_class][0]

        # Anzeige des Vorhersageergebnisses auf dem Videostream
        cv2.putText(frame, f"Vorhersage: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Video Stream', frame)

        # Beenden mit der Taste 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Lade das Modell und starte den Video-Stream
model = tf.keras.models.load_model("trained_model.h5")
predict_from_video_stream(model, class_dict)
