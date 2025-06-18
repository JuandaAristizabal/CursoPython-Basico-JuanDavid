import numpy as np

# Convertir grados Fahrenheit a Celsius
Temperature_fahrenheit = np.linspace(80,300,20)  # Temperaturas en Fahrenheit
print("Temperaturas en Farenheit:", Temperature_fahrenheit)

Temperature_celsius = (Temperature_fahrenheit - 32) * 5 / 9  # Fórmula de conversión
print("Temperaturas en Celsius:", Temperature_celsius)
