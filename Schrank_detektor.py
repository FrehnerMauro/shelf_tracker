import cv2
from ultralytics import YOLO

# threshold personen ekrennen
threshold = 0.5

# Lade das YOLO-Modell
model = YOLO('yolov8m.pt')

# Initialisiere den Zähler für Obst und Wasserflaschen
inventory = {"apple": 0, "bottle": 0}

# manuelle ROI setzen
x, y, width, height = 400, 750, 400, 150
#x, y, width, height = 400, 1000, 400, 130
#x, y, width, height = 400, 1200, 400, 130


def __main__():
    # Initialisiere die Webcam
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("/Users/maurofrehner/Desktop/shelfV2.mp4")
    #cap = cv2.VideoCapture("shelf1.mp4")
    if not cap.isOpened():
        print("Fehler beim Zugriff auf die Webcam.")
        exit()

    """
    ret, frame = cap.read()
    out = getRoisOfShelfs(frame)
    cv2.imshow("canny",out)
    cv2.waitKey(0)
    """

    while True:
        # Erfasse das Bild von der Webcam
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if isValid(frame):
            if not ret:
                print("Fehler beim Erfassen des Webcam-Bildes.")
                break
            
            #rois = getRoisOfShelfs()
            #for roi in rois:
                #doYolo(roi)


            # ROI extrahieren
            frame = extract_roi_from_frame(frame, x, y, width, height)
            
            result = doYolo(frame)
            #updateInventar()
            drawObjects(result, frame)
            

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
    res = doYolo(frame)
    for result in res:
        for box in result.boxes:
            cls = int(box.cls[0])  # Klassenindex extrahieren
            conf = float(box.conf[0])  # Konfidenzscore extrahieren
            if cls == 0 and conf >= threshold:  # Klasse "Person" und Schwellenwert prüfen
                return False
    return True
    

def doYolo(frame):
    results = model(frame)
    return results

def getBottleCount():
    pass

def updateInventar(results):
    """
            # Zeige den aktuellen Bestand auf dem Frame an
        cv2.putText(frame, f'Apple Count: {inventory["apple"]}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(frame, f'Bottle Count: {inventory["bottle"]}', (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                        # Wenn das erkannte Objekt ein Apfel oder eine Wasserflasche ist,q erhöhe den entsprechenden Zähler
                    if label == "apple":
                        inventory["apple"] += 1
                    elif label == "bottle":
                        inventory["bottle"] += 1
                        
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

                        
    """
    pass

def drawObjects(results, frame):
    for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    # Zeichne Rechteck und Beschriftung
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    return frame
    #return image with drwaings
    
    

def extract_roi_from_frame(frame, x, y, width, height):
    """
    Extrahiert eine Region of Interest (ROI) aus einem gegebenen Frame basierend auf den angegebenen Parametern.

    :param frame: Eingabeframe als numpy Array.
    :param x: X-Koordinate der oberen linken Ecke des ROI.
    :param y: Y-Koordinate der oberen linken Ecke des ROI.
    :param width: Breite des ROI.
    :param height: Höhe des ROI.
    :return: Ausgeschnittenes ROI als Bild.
    """
    # Überprüfen, ob die Koordinaten innerhalb des Frames liegen
    if x < 0 or y < 0 or x + width > frame.shape[1] or y + height > frame.shape[0]:
        raise ValueError("Die angegebenen Koordinaten liegen außerhalb des Frames.")

    # ROI aus dem Frame ausschneiden
    roi = frame[y:y+height, x:x+width]
    return roi



    
if __name__ == "__main__":
    __main__()
    