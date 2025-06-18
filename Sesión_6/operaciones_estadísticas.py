import numpy as np

# Crear una array de 0 a 5000 con 50 valores
valores = np.linspace(0, 5000, 50)
print("Promedio:", np.mean(valores))  # Calcular el promedio
print("Mediana:", np.median(valores))  # Calcular la mediana
print("Varianza:", np.var(valores))  # Calcular la varianza
print("Desviación estándar:", np.std(valores))  # Calcular la desviación estándar
print("Cuartiles:", np.percentile(valores, [25, 50, 75]))  # Calcular cuartiles
print("Máximo:", np.max(valores))  # Calcular el valor máximo
print("Mínimo:", np.min(valores))  # Calcular el valor mínimo
print("Suma:", np.sum(valores))  # Calcular la suma
print("Desviación media:", np.mean(np.abs(valores - np.mean(valores))))  # Calcular la desviación media
print("Rango:", np.ptp(valores))  # Calcular el rango (diferencia entre máximo y mínimo)
print("Coeficiente de variación:", np.std(valores) / np.mean(valores))  # Calcular el coeficiente de variación

