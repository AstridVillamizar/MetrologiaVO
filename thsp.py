import os
import cv2
import numpy as np

datafolder = './Poxipol/'
B = np.zeros((484, 1), dtype=np.uint8)

for i in range(11516, 12000):
    img_path = os.path.join(datafolder, f'Pox_{i}.bmp')
    a = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Se toma la region de interes del patron de speckle
    A = a[169:191, 169:191]  
    
    # Convertir la región en un vector
    A = A.flatten()  # convertir matriz en un vector
    
    # Agregar el vector como una columna a la matriz B
    B = np.concatenate((B, A[:, np.newaxis]), axis=1)

# Eliminar la primera columna de ceros de B
B = B[:, 1:]

# Visualizar la imagen a partir de la matriz B (unir los vectores en una imagen)
B_img = np.reshape(B, (484, 484)).astype(np.uint8)
cv2.imshow('Imagen B', B_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar la imagen resultante como thsp0.bmp
cv2.imwrite('thsp0.bmp', B_img)
