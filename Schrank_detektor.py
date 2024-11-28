import cv2
from ultralytics import YOLO

# threshold personen ekrennen
threshold = 0.5

# Lade das YOLO-Modell
model = YOLO('yolov8m.pt')

# Initialisiere den Zähler für Obst und Wasserflaschen
inventory = {"apple": 0, "bottle": 0}




def __main__():
    
    frame_count = 0
    skip_frames = 4
    
    inventory = {"apple": 0, "bottle": 0}

    
    # Initialisiere das Bild
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("/Users/maurofrehner/Desktop/shelfV2.mp4")
    #cap = cv2.VideoCapture(r"C:\Users\marku\Documents\StudiumMobileRobotics\7.Semester\Bildverarbeitung2\Projekt\shelfV2.mp4")
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
        
        if not ret:  # Beende die Schleife, wenn kein Frame mehr gelesen werden kann
            print("Kein Frame verfügbar. Beende...")
            break

        # Überspringe Frames basierend auf `skip_frames`
        if frame_count < skip_frames:
            frame_count += 1
            continue
        else:
            frame_count = 0  # Frame-Zähler zurücksetzen


        
        # Verarbeite das aktuelle Frame
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


            


        # ROIs extrahieren
        frame1, frame2, frame3, frame4 = extract_roi_from_frame(frame)

        # YOLO-Verarbeitung
        result1 = doYolo(frame1)
        result2 = doYolo(frame2)
        result3 = doYolo(frame3)
        result4 = doYolo(frame4)

        # Ergebnisse zeichnen
        drawObjects(result1, frame1)
        drawObjects(result2, frame2)
        drawObjects(result3, frame3)
        drawObjects(result4, frame4)
        
                # prüfe auf person
        if isValid(frame):
            inventory["apple"] = 0
            inventory["bottle"] = 0
            # Inventar aktualisieren
            updateInventar(result1, inventory)
            updateInventar(result2, inventory)
            updateInventar(result3, inventory)
            updateInventar(result4, inventory)
        else:
            # Schreibe Text ins Bild
            cv2.putText(frame, "Person erkannt!", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            
        # Inventar anzeigen
        draw_inventory_on_frame(inventory, frame)
        
        # Zeige das verarbeitete Bild und die ROIs an
        cv2.imshow("YOLO Object Detection", frame)
        cv2.imshow("Regal 1", frame1)
        cv2.imshow("Regal 2", frame2)
        cv2.imshow("Regal 3", frame3)
        cv2.imshow("Regal 4", frame4)

        # Beende das Programm bei Tastendruck
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
    #results = model(frame)
    results = model(frame, classes=[0, 39, 47]) # mit Maske für person und Apfel und Wasserflasche
    return results



def updateInventar(yolo_results_list, inventory):
    for yolo_results in yolo_results_list:
        class_names = yolo_results.names
        boxes = yolo_results.boxes

        for box in boxes:
            class_id = int(box.cls)
            label = class_names[class_id]

            if label == 'apple':
                inventory["apple"] += 1
            elif label == 'bottle':
                inventory["bottle"] += 1

"""
    # Schreibe das Inventar ins Frame
    y_offset = 20
    for key, value in inventory.items():
        cv2.putText(frame, f'{key.capitalize()} Count: {value}', (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        y_offset += 30

    # Debugging: Ausgabe des aktualisierten Inventars
    print(f"Aktualisiertes Inventar: {inventory}")

"""

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
    
    

def extract_roi_from_frame(frame):

    # manuelle ROI setzen
    x1, y1, width1, height1 = 400, 520, 400, 150
    x2, y2, width2, height2 = 400, 750, 400, 150
    x3, y3, width3, height3 = 400, 1000, 400, 130
    x4, y4, width4, height4 = 400, 1250, 400, 150


    # Überprüfen, ob die Koordinaten innerhalb des Frames liegen
    if x1 < 0 or y1 < 0 or x1 + width1 > frame.shape[1] or y1 + height1 > frame.shape[0]:
        raise ValueError("Die angegebenen Koordinaten liegen außerhalb des Frames.")

    # ROI aus dem Frame ausschneiden
    roi1 = frame[y1:y1+height1, x1:x1+width1]
    roi2 = frame[y2:y2+height2, x2:x2+width2]
    roi3 = frame[y3:y3+height3, x3:x3+width3]
    roi4 = frame[y4:y4+height4, x4:x4+width4]

    return roi1, roi2, roi3, roi4



def draw_inventory_on_frame(inventory, frame):

    y_offset = 30  # Zeilenhöhe zwischen den Texten
    x, y = 10, 30  # Startposition für den Text (oben links)
    
    for key, value in inventory.items():
        text = f"{key}: {value}"
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += y_offset

    return frame


    
if __name__ == "__main__":
    __main__()
    