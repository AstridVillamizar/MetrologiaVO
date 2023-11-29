import matplotlib.pyplot as plt
from laguerre_gauss import LaguerreGauss
from residuos import Residuos
from subpix2x2_s import subpix2x2_s
import numpy as np

# Aquí va el código para usar estas funciones, como leer la imagen y llamar a las funciones
transmittanceImage = plt.imread('thsp0.bmp')
Ima = transmittanceImage
complex1 = LaguerreGauss(Ima)
t = Residuos(complex1)
cpos = np.sum(t == 1)
CV = cpos 
T = subpix2x2_s(complex1, t)
print('el numero de vortices es', CV)
