import numpy as np

def LaguerreGauss(transmittanceImage):
    omega = 0.02 #ancho del filtro LG 
    transmittanceImage = np.array(transmittanceImage)
    m, n = transmittanceImage.shape #dimensiones de la imagen
    fresnelTransform2 = transmittanceImage
    #transformada de Fourier
    fresnelTransform2 = np.fft.fft2(fresnelTransform2, s=(m, n))
    fresnelTransform2 = np.fft.fftshift(fresnelTransform2)

    #vectores k y l para manipular el filtro LG
    k = np.arange(1, m + 1)
    kk = np.arange(m, 0, -1)
    K = (k - kk) / 2

    l = np.arange(1, n + 1)
    ll = np.arange(n, 0, -1)
    L = (l - ll) / 2
    L = np.transpose([L])

    U, V = np.meshgrid(K, L) #mallas de coordenadas
    #calculo del filtro LG con nucleo circular
    omega2 = omega ** 2
    omega4 = omega ** 4
    pi2 = np.pi ** 2
    opi = pi2 * omega2
    lg = 1j * pi2 * (omega4) * (V + 1j * U) * np.exp(-opi * ((V ** 2) + (U ** 2)))
    LG = np.fft.fft2(lg, s=(m, n))
    LG = np.fft.fftshift(LG)
    #operacion de pseudofase
    pseudofas = (LG * fresnelTransform2)
    psfas = np.fft.ifft2(pseudofas, s=(m, n))
    psfas = np.fft.fftshift(psfas)
    a = psfas

    bb1 = np.real(psfas) / np.max(np.max(np.real(psfas)))
    #plt.imshow(bb1, cmap='gray')
    #plt.imsave('pfase.png', bb1, cmap='gray')
     
    #proceso para manipular la fase
    complex = np.zeros((m, n), dtype=np.complex128)
    for row in range(m):
        for column in range(n):
            if row % 2 != 0:
                if column % 2 == 0:
                    complex[row, column] = a[row, column] * -1
                else:
                    complex[row, column] = a[row, column]
            else:
                if column % 2 != 0:
                    complex[row, column] = a[row, column] * -1
                else:
                    complex[row, column] = a[row, column]

    return complex