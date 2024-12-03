import cv2
import numpy as np
import random
from ultralytics import YOLO

# YOLO Params
model = YOLO('yolov8m.pt')
threshold = 0.5 # threshold personen ekrennen

# Initialisiere den Zähler für Obst und Wasserflaschen
inventory = {"apple": 0, "bottle": 0}

#parms for get Rois
debugGetRois = False
mult = 25 #for aproximate contours
sollRoiCount = 4
perimeterMin = 1300
perimeterMax = 5000
cannyThresholdLower = 30
cannyThresholdUper = 60

def __main__():
    frame_count = 0
    skip_frames = 4
    getRoisSuccessfull = False
    rois = []
    inventory = {"apple": 0, "bottle": 0}
    
    #cap = cv2.VideoCapture("/Users/maurofrehner/Desktop/shelfV2.mp4")
    #cap = cv2.VideoCapture(r"C:\Users\marku\Documents\StudiumMobileRobotics\7.Semester\Bildverarbeitung2\Projekt\shelfV2.mp4")
    cap = cv2.VideoCapture("shelfV2.mp4")
    if not cap.isOpened():
        print("Fehler beim Zugriff auf die Webcam.")
        exit()

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
            
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) # Verarbeite das aktuelle Frame
        
        if not getRoisSuccessfull and isValid(frame):
            getRoisSuccessfull, rois = getRoisOfShelfs(frame)  
            
        elif getRoisSuccessfull and isValid(frame):
            inventory["apple"] = 0
            inventory["bottle"] = 0

            roiFrames = extractRoiFrames(frame,rois)
            
            for idx in range(0,len(roiFrames)):
                result = doYolo(roiFrames[idx])
                drawObjects(result, roiFrames[idx])
                updateInventar(result, inventory)
                cv2.imshow("Regal "+str(idx), roiFrames[idx])
                
        else:
            cv2.putText(frame, "Person erkannt!", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
        draw_inventory_on_frame(inventory, frame)
        cv2.imshow("YOLO Object Detection", scaleImage(frame))
        
        # Beende das Programm bei Tastendruck
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Freigeben der Ressourcen
    cap.release()
    cv2.destroyAllWindows()
    
def getRoisOfShelfs(frame):
    rois = []

    # Perform Canny edge detection
    canny = cv2.Canny(frame, threshold1=cannyThresholdLower, threshold2=cannyThresholdUper)
    kernel = np.ones((5,5),np.uint8)
    canny = cv2.dilate(canny,kernel,iterations = 3)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    all_contours, __ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contourFrame = frame.copy()
    for contour in all_contours:
        # Approximate contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.002* mult * perimeter, True)

        # Calculate aspect ratio and bounding box
        if perimeter >= perimeterMin and perimeter <= perimeterMax and len(approx) >=4:
            x,y,w,h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if aspect_ratio > 1.5 and aspect_ratio < 2.8:
                rois.append([x,y,w,h])
                if debugGetRois:
                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h), color=(255,0,0),thickness=4)
            if debugGetRois:
                cv2.drawContours(contourFrame, [approx], -1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 3)
                cv2.putText(contourFrame,"Rectangle",(x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2,)
        
    if debugGetRois:
        cv2.imshow("Canny",scaleImage(canny))
        cv2.imshow("contour",scaleImage(contourFrame))
        cv2.imshow("roi",scaleImage(frame))
        
    if len(rois) == sollRoiCount:
        return True, rois 
    return False, rois

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

def extractRoiFrames(frame, rois):

    # manuelle ROI setzen
    x1, y1, width1, height1 = rois[0][0], rois[0][1], rois[0][2], rois[0][3]
    x2, y2, width2, height2 = rois[1][0], rois[1][1], rois[1][2], rois[1][3]
    x3, y3, width3, height3 = rois[2][0], rois[2][1], rois[2][2], rois[2][3]
    x4, y4, width4, height4 = rois[3][0], rois[3][1], rois[3][2], rois[3][3]


    # Überprüfen, ob die Koordinaten innerhalb des Frames liegen
    if x1 < 0 or y1 < 0 or x1 + width1 > frame.shape[1] or y1 + height1 > frame.shape[0]:
        raise ValueError("Die angegebenen Koordinaten liegen außerhalb des Frames.")

    # ROI aus dem Frame ausschneiden
    roiFrames=[]
    roiFrames.append(frame[y1:y1+height1, x1:x1+width1])
    roiFrames.append(frame[y2:y2+height2, x2:x2+width2])
    roiFrames.append(frame[y3:y3+height3, x3:x3+width3])
    roiFrames.append(frame[y4:y4+height4, x4:x4+width4])

    return roiFrames

def draw_inventory_on_frame(inventory, frame):

    y_offset = 30  # Zeilenhöhe zwischen den Texten
    x, y = 10, 30  # Startposition für den Text (oben links)
    
    for key, value in inventory.items():
        text = f"{key}: {value}"
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += y_offset

    return frame

def scaleImage(img, scale=2.0):
    width = int(img.shape[1]  / scale)
    height = int(img.shape[0]  / scale)
    return cv2.resize(img,(width, height))
    
if __name__ == "__main__":
    __main__()
    