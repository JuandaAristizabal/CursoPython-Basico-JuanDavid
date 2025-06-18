import numpy as np
import matplotlib.pyplot as plt
import time as time

# Crear una lista de presiones en psi
presiones = np.array([1013, 1015, 1012, 1010, 1008, 1005, 1003, 1001])
print(presiones)

presiones = presiones + 14.4
print(presiones)

profundidades = np.linspace(0, 3000, 10) # metros
presion_superficie = 1200 # psia
gradiente = 0.35 # psia/m

presion_pozo = presion_superficie - gradiente * profundidades
print("Presion a diferentes profundidades (psia):")
print(presion_pozo)

print("Presión máxima:", np.max(presion_pozo)) 
print("Presión mínima:", np.min(presion_pozo)) 
print("Promedio:", np.mean(presion_pozo)) 
print("Desviación estándar:", np.std(presion_pozo)) 

# Crear un array de presiones en psi 
lista = [1000, 950, 875, 840] 
presiones = np.array(lista) 
print("Presiones:", presiones) 
print("Dimensiones:", presiones.ndim) 
print("Forma:", presiones.shape) 
print("Tipo de dato:", presiones.dtype)

arreglo3d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("Promedio por columnas:" , np.mean(arreglo3d, axis=0)) # Promedio por columnas 
print("Promedio por Filas:" , np.mean(arreglo3d, axis=1)) # Promedio por filas

print(arreglo3d[0:2]) # Primeras dos filas
print(arreglo3d[0:2, 1:3]) # Primeras dos filas, columnas 1 y 2

print(np.zeros((3, 4))) # Crear un array de ceros de 3x4
print(np.ones((2, 3))) # Crear un array de unos de 2x3
print(np.full((2, 2), 7)) # Crear un array de 2x2 lleno de 7
print(np.eye(3)) # Crear una matriz identidad de 3x3

unos = np.ones((3, 3))
print(arreglo3d + unos) # Sumar posición en arreglo3d + posición en array unos
print(arreglo3d - unos) # Restar posición en arreglo3d - posición en array unos
print(arreglo3d * unos) # Multiplicar posición en arreglo3d * posición en array unos
print(arreglo3d / unos) # Dividir posición en arreglo3d / posición en array unos

print(arreglo3d.shape)  # Dimensiones del arreglo
print(len(arreglo3d)) # Número de filas
print(arreglo3d.size) # Número total de elementos
print(arreglo3d.ndim) # Número de dimensiones
print(arreglo3d.reshape(1, 9)) # Cambiar forma a 1 fila y 9 columnas
print(arreglo3d.dtype) # Tipo de dato del arreglo

arreglo3d2 = arreglo3d * 2
print(arreglo3d == arreglo3d2) # Comparar si son iguales


