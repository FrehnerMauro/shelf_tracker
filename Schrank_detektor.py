import cv2
from ultralytics import YOLO



# Lade das YOLO-Modell
model = YOLO("yolov5s.pt")

def __main__():
    # Initialisiere die Webcam
    cap = cv2.VideoCapture("shelf1.mp4")
    if not cap.isOpened():
        print("Fehler beim Zugriff auf die Webcam.")
        exit()

    # Initialisiere den Zähler für Obst und Wasserflaschen
    inventory = {"apple": 0, "bottle": 0}

    ret, frame = cap.read()
    out = getRoisOfShelfs(frame)
    cv2.imshow("canny",out)
    cv2.waitKey(0)

    while True:
        # Erfasse das Bild von der Webcam
        ret, frame = cap.read()
        if isValid(frame):
            if not ret:
                print("Fehler beim Erfassen des Webcam-Bildes.")
                break

            rois = getRoisOfShelfs()
            for roi in rois:
                doYolo(roi)
                
            updateInventar()
            drawObjects()
            
        ### old stuff move to according functions###
            # Vorhersagen mit YOLO-Modell
            results = model(frame)

            # Setze die Zähler zurück für diesen Frame
            inventory["apple"] = 0
            inventory["bottle"] = 0

            # Zeichnen der Begrenzungsrahmen und Beschriftungen auf dem Frame
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    # Wenn das erkannte Objekt ein Apfel oder eine Wasserflasche ist,q erhöhe den entsprechenden Zähler
                    if label == "apple":
                        inventory["apple"] += 1
                    elif label == "bottle":
                        inventory["bottle"] += 1

                    # Zeichne Rechteck und Beschriftung
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

        # Zeige den aktuellen Bestand auf dem Frame an
        cv2.putText(frame, f'Apple Count: {inventory["apple"]}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(frame, f'Bottle Count: {inventory["bottle"]}', (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        # Zeige das verarbeitete Bild an
        cv2.imshow("YOLO Object Detection", frame)

        # Beenden, wenn 'q' gedrückt wird
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Freigeben der Ressourcen
    cap.release()
    cv2.destroyAllWindows()
    
def getRoisOfShelfs(frame):
    blurred_image = cv2.GaussianBlur(frame, (5, 5), 0)
    # Perform Canny edge detection
    return cv2.Canny(blurred_image, threshold1=100, threshold2=200)

def isValid(frame):
    #person in frame?
    return True

def doYolo(frame):
    pass
    #return obect list

def getBottleCount():
    pass

def updateInventar():
    pass

def drawObjects():
    pass
    #return image with drwaings