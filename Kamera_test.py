import cv2

# Kamera öffnen (0 für die Standardkamera, ändere dies für andere Kameras)
cap = cv2.VideoCapture(1)

# Überprüfen, ob die Kamera erfolgreich geöffnet wurde
if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden.")
    exit()

# Auflösung einstellen (z. B. 640x480)
frame_width = 1920
frame_height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Framerate (FPS) einstellen (z. B. 30 FPS)
fps = 10
cap.set(cv2.CAP_PROP_FPS, fps)

# Überprüfen, ob die Einstellungen erfolgreich übernommen wurden
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Auflösung eingestellt: {int(actual_width)}x{int(actual_height)}")
print(f"Framerate eingestellt: {actual_fps} FPS")

# Hauptschleife zum Capturen und Anzeigen des Kamerabildes
while True:
    # Ein Bild von der Kamera lesen
    ret, frame = cap.read()

    # Wenn das Bild erfolgreich geladen wurde
    if not ret:
        print("Fehler: Kein Bild erhalten.")
        break

    # Bild anzeigen
    cv2.imshow("Kamera Bild", frame)

    # Warten auf eine Taste (z.B. 'q' zum Beenden)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera freigeben und alle Fenster schließen
cap.release()
cv2.destroyAllWindows()
