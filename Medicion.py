import cv2
import numpy as np
from Detector_Objetos import *

# Cargamos el detector del marcador del ARUCO
parametros = cv2.aruco.DetectorParameters()
diccionario = cv2.aruco.Dictionary(cv2.aruco.DICT_5X5_100,5)

# Cargamos el detector de objetos
detector = DetectorFondoHomogeneo()

# Realizamos la videocaptura de nuestra cámara
cap = cv2.VideoCapture(1)
cap.set(34, 640)
cap.set(4, 480)

# Accedemos al while principal
while True:
    # Realizamos la lectura de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # MARCADOR ARUCO
    # Detectamos el marcador ArUco
    # Página para creación de ArUco -> chev.me/arucogen/
    # Página para documentación sobre las librerías del ArUco docs.opencv.org/4.5.2/d5/dae/tutorial_aruco_detection.html
    esquinas, _, _ = cv2.aruco.detectMarkers(frame, diccionario, parameters=parametros)

    # Verificamos si se detectaron marcadores
    if len(esquinas) > 0:
        esquinas_ent = np.int32(esquinas)
        cv2.polylines(frame, esquinas_ent, True, (0, 0, 255), 5)

        # Perímetro del ArUco
        perimetro_aruco = cv2.arcLength(esquinas_ent[0], True)

        # Proporción en centímetros
        proporcion_cm = perimetro_aruco / 16  # El número depende del ArUco
    else:
        # Si no se detecta ningún marcador, establecer una proporción predeterminada
        proporcion_cm = 1

    # Detección de objetos
    # Detectamos los objetos
    contornos = detector.deteccion_objetos(frame)

    # Dibujamos la detección del objeto
    for cont in contornos:
        # Rectángulo del objeto
        # A partir del polígono anterior vamos a obtener un rectángulo
        rectangulo = cv2.minAreaRect(cont)
        (x, y), (an, al), angulo = rectangulo

        # Pasamos el ancho y el alto de píxeles a centímetros
        ancho = an / proporcion_cm
        alto = al / proporcion_cm

        # Dibujamos un círculo en la mitad del rectángulo
        cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)

        # Vamos a dibujar el rectángulo que ya obtuvimos
        rect = cv2.boxPoints(rectangulo)  # Obtenemos el rectángulo
        rect = np.int32(rect)  # Aseguramos que toda la información esté en enteros

        # Dibujamos el rectángulo
        cv2.polylines(frame, [rect], True, (0, 255, 0), 2)

        # Mostramos la información en píxeles
        cv2.putText(frame, "Ancho: {}cm".format(round(ancho, 1)), (int(x), int(y - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 0, 255), 2)
        cv2.putText(frame, "Alto: {}cm".format(round(alto, 1)), (int(x), int(y + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (75, 0, 75), 2)

    # Mostramos los fotogramas
    cv2.imshow('Medición de Objetos', frame)
    # Si le damos ESC, salimos del programa
    t = cv2.waitKey(1)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
