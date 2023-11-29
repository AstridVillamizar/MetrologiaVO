import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def subpix2x2_s(complex, residuos):
    #inicio y dimensiones
    DFT = complex
    m, n = DFT.shape
    l = 0
    PV = np.zeros((3, 3, m * n), dtype=np.complex_)
    PLV = []

    #ciclo para extraer submatrices importantes
    for i in range(1, m - 1):
        for j in range (1, n - 1):
            if residuos[i - 1, j - 1] and i > 0 and j > 0:
                PV[:, :, l] = DFT[i - 1:i + 2, j - 1:j + 2]
                l += 1
                PLV.append([j, i])

    PLV = np.array(PLV).T
    # separacion de parte real e imaginaria
    rvortp = np.real(PV[:, :, :l])
    ivortp = np.imag(PV[:, :, :l])
    # creacion de matrices para manipular
    RPVort = np.zeros((9, 3, l))
    IPVort = np.zeros((9, 3, l))

    k = 0
    for i in range(1, 4):
        for j in range(1, 4):
            RPVort[k, 2, :] = rvortp[i - 1, j - 1, :]
            RPVort[k, 1, :] = i
            RPVort[k, 0, :] = j
            IPVort[k, 2, :] = ivortp[i - 1, j - 1, :]
            IPVort[k, 1, :] = i
            IPVort[k, 0, :] = j
            k += 1
    #calculos entre matrices y vectores
    A = np.zeros((3, 3))
    for i in range(9):
        A[0, 0] += RPVort[i, 0, 0] ** 2
        A[0, 1] += RPVort[i, 0, 0] * RPVort[i, 1, 0]
        A[0, 2] += RPVort[i, 0, 0]
        A[1, 1] += RPVort[i, 1, 0] ** 2
        A[1, 2] += RPVort[i, 1, 0]
    A[1, 0] = A[0, 1]
    A[2, 0] = A[0, 2]
    A[2, 1] = A[1, 2]
    A[2, 2] = 9

    bp = np.zeros((3, l))
    ddp = np.zeros((3, l))
    for w in range(l):
        for i in range(9):
            bp[:2, w] += RPVort[i, :2, w] * RPVort[i, 2, w]
            ddp[:2, w] += IPVort[i, :2, w] * IPVort[i, 2, w]
    #solucion de sistema de ecuaciones
    PVR = np.zeros((3, l))
    PVI = np.zeros((3, l))
    for w in range(l):
        PVR[:, w] = np.linalg.solve(A, bp[:, w])
        PVI[:, w] = np.linalg.solve(A, ddp[:, w])

    MEQ = np.zeros((2, 2, l))
    VEQ = np.zeros((2, l))
    for w in range(l):
        MEQ[:, :, w] = np.vstack((PVR[:2, w], PVI[:2, w]))
        VEQ[:, w] = -1 * np.array([PVR[2, w], PVI[2, w]])

    SLV = np.zeros((2, l))
    for w in range(l):
        SLV[:, w] = np.linalg.solve(MEQ[:, :, w], VEQ[:, w])
    #coordenadas en el plano
    SPLV = SLV + PLV - 1
    VVX = SPLV[0, :]
    VVY = SPLV[1, :]
    #calculo de las propiedades estructurales
    l = MEQ.shape[2]
    e = np.zeros(l)
    theta = np.zeros(l)
    omega = np.zeros(l)
    q = np.zeros(l)
    for w in range(l):
        e[w] = np.sqrt(1 - (((MEQ[0, 0, w] ** 2) + (MEQ[1, 0, w] ** 2) + (MEQ[0, 1, w] ** 2) + (MEQ[1, 1, w] ** 2)) - np.sqrt(((MEQ[0, 0, w] ** 2) + (MEQ[1, 0, w] ** 2) - (MEQ[0, 1, w] ** 2) - (MEQ[1, 1, w] ** 2)) ** 2 + 4 * ((MEQ[0, 0, w] * MEQ[0, 1, w]) + (MEQ[1, 0, w] * MEQ[1, 1, w])) ** 2)) / ((MEQ[0, 0, w] ** 2) + (MEQ[1, 0, w] ** 2) + (MEQ[0, 1, w] ** 2) + (MEQ[1, 1, w] ** 2) + np.sqrt(((MEQ[0, 0, w] ** 2) + (MEQ[1, 0, w] ** 2) - (MEQ[0, 1, w] ** 2) - (MEQ[1, 1, w] ** 2)) ** 2 + 4 * ((MEQ[0, 0, w] * MEQ[0, 1, w]) + (MEQ[1, 0, w] * MEQ[1, 1, w])) ** 2)))
        theta[w] = np.abs(np.arctan((MEQ[0, 0, w] * MEQ[1, 1, w] - MEQ[1, 0, w] * MEQ[0, 1, w]) / (MEQ[0, 0, w] * MEQ[1, 0, w] + MEQ[0, 1, w] * MEQ[1, 1, w])))
        if theta[w] >= np.pi / 2:
            theta[w] = np.pi - theta[w]
        omega[w] = np.abs(MEQ[0, 0, w] * MEQ[1, 1, w] - MEQ[1, 0, w] * MEQ[0, 1, w])
        q[w] = np.sign(MEQ[0, 0, w] * MEQ[1, 1, w] - MEQ[1, 0, w] * MEQ[0, 1, w])

    S0 = np.zeros(l)
    S1 = np.zeros(l)
    S2 = np.zeros(l)
    S3 = np.zeros(l)
    for w in range(l):
        S0[w] = (MEQ[0, 0, w] ** 2) + (MEQ[0, 1, w] ** 2) + (MEQ[1, 0, w] ** 2) + (MEQ[1, 1, w] ** 2)
        S1[w] = (MEQ[0, 0, w] ** 2) + (MEQ[1, 0, w] ** 2) - (MEQ[0, 1, w] ** 2) - (MEQ[1, 1, w] ** 2)
        S2[w] = 2 * (MEQ[0, 0, w] * MEQ[0, 1, w] + MEQ[1, 0, w] * MEQ[1, 1, w])
        S3[w] = 2 * (MEQ[0, 0, w] * MEQ[1, 1, w] - MEQ[1, 0, w] * MEQ[0, 1, w])
    # Cálculo de parámetros de Poincaré (S0, S1, S2, S3)
    S = np.vstack((S1 / S0, S2 / S0, S3 / S0)).T

    # Guardando los resultados en un archivo Excel Propiedades y coordenadas 3D
    pd.DataFrame({'e': e, 'theta': theta, 'omega': omega, 'q': q}).to_excel('results.xlsx', index=False)
    pd.DataFrame({'S1': S1, 'S2': S2, 'S3': S3}).to_excel('coordenadas.xlsx', index=False)
    


    # Creación y visualización de la esfera de Poincaré
    theta1, phi = np.linspace(0, 2*np.pi, 40), np.linspace(0, np.pi, 40)
    theta1, phi = np.meshgrid(theta1, phi)
    rho = 1
    x = rho * np.sin(phi) * np.cos(theta1)
    y = rho * np.sin(phi) * np.sin(theta1)
    z = rho * np.cos(phi)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Dibujar la esfera
    ax.plot_surface(x, y, z, color='gray', alpha=0.5, linewidth=0, antialiased=True)
    ax.set_box_aspect([1,1,1])  # Importante para una esfera no aplastada

    # Calcular la distancia de cada punto al centro (en términos del eje Z)
    distancias_al_centro = np.abs(S[:, 2])

    # Usar un mapa de colores para los puntos
    # Por ejemplo, 'coolwarm' tiene rojo para valores altos y azul para valores bajos
    puntos = ax.scatter(S[:, 0], S[:, 1], S[:, 2], c=distancias_al_centro, cmap='coolwarm', s=50)

    # Crear una barra de colores para interpretar los colores de los puntos
    cbar = fig.colorbar(puntos, ax=ax, shrink=0.6)
    cbar.set_label('Distancia al centro (eje Z)', fontsize=12)

    # Etiquetas y títulos
    ax.set_xlabel('S1', fontsize=12)
    ax.set_ylabel('S2', fontsize=12)
    ax.set_zlabel('S3', fontsize=12)
    ax.set_title('Visualización de la Esfera de Poincaré', fontsize=14)

    # Ajustar la perspectiva
    ax.view_init(elev=30, azim=120)

    plt.savefig('esfera.jpg')
    plt.show()