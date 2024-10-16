import cv2

class DetectorFondoHomogeneo():
    def __init__(self):
        pass

    def deteccion_objetos(self, frame):
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Creación de una máscara con un umbral adaptativo
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

        # Encontrar contornos
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Creación de una lista para almacenar los objetos
        objetos_contornos = []

        # Si encontramos contornos, entonces entramos al ciclo FOR
        for cnt in contornos:
            # Medimos el área de los contornos
            area = cv2.contourArea(cnt)
            # Si el área es mayor a 2000, se va a agregar el objeto a la lista
            if area > 2000:
                objetos_contornos.append(cnt)

        return objetos_contornos
