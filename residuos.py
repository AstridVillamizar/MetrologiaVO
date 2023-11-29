import numpy as np
import matplotlib.pyplot as plt

def Residuos(complex):
    #obtener matriz de pseudofase
    IM_phase = np.angle(complex)
    rows, cols = IM_phase.shape
    #crear matrices desplazadas para calcular los residuos
    IM_active = IM_phase.copy()
    IM_below = np.zeros((rows, cols))
    IM_below[:rows - 1, :] = IM_phase[1:rows, :]
    
    IM_right = np.zeros((rows, cols))
    IM_right[:, :cols - 1] = IM_phase[:, 1:cols]
    
    IM_belowright = np.zeros((rows, cols))
    IM_belowright[:rows - 1, :cols - 1] = IM_phase[1:rows, 1:cols]
    #calcular los residuos en diferentes direcciones
    res1 = np.mod(IM_active - IM_below + np.pi, 2 * np.pi) - np.pi
    res2 = np.mod(IM_below - IM_belowright + np.pi, 2 * np.pi) - np.pi
    res3 = np.mod(IM_belowright - IM_right + np.pi, 2 * np.pi) - np.pi
    res4 = np.mod(IM_right - IM_active + np.pi, 2 * np.pi) - np.pi
  
    #sumar los residuos y aplicar condiciones para identificar los vortices
    temp_residues = res1 + res2 + res3 + res4
    residues = (temp_residues >= 6)
    residues = residues ^ (temp_residues <= -6)

    #establecer cierta areas cercanas a los bordes como no vortices
    residues[:, cols - 1] = 0
    residues[rows - 1, :] = 0
    residues[:, 0] = 0
    residues[0, :] = 0
    residue_charge = residues
    #visualizacion de los vortices en la pseudofase
    alto, ancho = complex.shape
    plt.figure()
    plt.imshow(np.angle(complex), cmap='gray')


    for i in range(alto):
        for j in range(ancho):
            if residue_charge[i, j] == 1:
                plt.plot(j, i, '*g')
                plt.axis('square')

    
    plt.savefig('vortices0.jpg')
    plt.show()

    return residue_charge