import numpy as np  

# Datos de producción de petróleo en diferentes pozos
[120, 130, 125, 140, 135, 128, 132],  # Pozo 1
[110, 115, 118, 120, 122, 119, 121],  # Pozo 2
[150, 155, 160, 158, 162, 159, 161]   # Pozo 3

#Transformar los datos en un array de NumPy
produccion_pozos = np.array([
    [120, 130, 125, 140, 135, 128, 132],  # Pozo 1
    [110, 115, 118, 120, 122, 119, 121],  # Pozo 2
    [150, 155, 160, 158, 162, 159, 161]   # Pozo 3
])

# Asignar cada pozo a una variable
pozo1 = produccion_pozos[0]
pozo2 = produccion_pozos[1]
pozo3 = produccion_pozos[2]

# Calcular producción total por pozo
produccion_total_Pozo1 = print("Producción Total Pozo 1" , np.sum(pozo1), "barriles") 
produccion_total_Pozo2 = print("Producción Total Pozo 2" , np.sum(pozo2), "barriles")
produccion_total_Pozo3 = print("Producción Total Pozo 3" , np.sum(pozo3), "barriles")
produccion_total_pozos = print("Producción Total de todos los pozos:", np.sum(produccion_pozos), "barriles")

# Precio del barril de petróleo
precio_petroleo = np.array([78, 82, 85, 80, 79, 83, 84])

# Dias de la semana
dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

# Calcular ingresos totales por pozo
ingresos_pozo1 = np.sum(pozo1 * precio_petroleo)
ingresos_pozo2 = np.sum(pozo2 * precio_petroleo)
ingresos_pozo3 = np.sum(pozo3 * precio_petroleo)

# Calcular ingresos totales de todos los pozos
ingresos_totales = np.sum(produccion_pozos * precio_petroleo)

# Imprimir resultados
print("Ingresos Totales Pozo 1:", ingresos_pozo1, "USD")
print("Ingresos Totales Pozo 2:", ingresos_pozo2, "USD")
print("Ingresos Totales Pozo 3:", ingresos_pozo3, "USD")
print("Ingresos Totales de todos los pozos:", ingresos_totales, "USD")

# Identificar el array de pozos con los dias de la semana
dias_semana = np.array(["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])

# Identificar el día con mayor ingreso por pozo
ingresos_por_dia = np.sum(produccion_pozos * precio_petroleo, axis=0)
dia_max_ingresos_pozo1 = np.argmax(pozo1 * precio_petroleo)
dia_max_ingresos_pozo2 = np.argmax(pozo2 * precio_petroleo)
dia_max_ingresos_pozo3 = np.argmax(pozo3 * precio_petroleo)

nombre_pozo1_max_ganancia = dias_semana[dia_max_ingresos_pozo1]
nombre_pozo2_max_ganancia = dias_semana[dia_max_ingresos_pozo2]
nombre_pozo3_max_ganancia = dias_semana[dia_max_ingresos_pozo3]

print(f"Día con más ingresos en Pozo 1: {nombre_pozo1_max_ganancia} con ingresos de {pozo1[dia_max_ingresos_pozo1] * precio_petroleo[dia_max_ingresos_pozo1]} USD")
print(f"Día con más ingresos en Pozo 2: {nombre_pozo2_max_ganancia} con ingresos de {pozo2[dia_max_ingresos_pozo2] * precio_petroleo[dia_max_ingresos_pozo2]} USD")
print(f"Día con más ingresos en Pozo 3: {nombre_pozo3_max_ganancia} con ingresos de {pozo3[dia_max_ingresos_pozo3] * precio_petroleo[dia_max_ingresos_pozo3]} USD")

# Calcular el dia que mas ingresos se obtuvo
dia_max_ingresos = np.argmax(np.sum(produccion_pozos * precio_petroleo, axis=0))
nombre_dia_max_ganancia = dias_semana[dia_max_ingresos]  
print(f"Día con más ingresos: {nombre_dia_max_ganancia}")  
print("Ingresos en ese día:", np.sum(produccion_pozos * precio_petroleo, axis=0)[dia_max_ingresos], "USD")

# Calcular el promedio de producción por pozo
promedio_produccion_pozo1 = np.mean(pozo1)
promedio_produccion_pozo2 = np.mean(pozo2)
promedio_produccion_pozo3 = np.mean(pozo3)

print("Promedio de producción Pozo 1:", promedio_produccion_pozo1, "barriles")
print("Promedio de producción Pozo 2:", promedio_produccion_pozo2, "barriles")
print("Promedio de producción Pozo 3:", promedio_produccion_pozo3, "barriles")  

# Calcular el promedio de ingresos por pozo
promedio_ingresos_pozo1 = np.mean(pozo1 * precio_petroleo)
promedio_ingresos_pozo2 = np.mean(pozo2 * precio_petroleo)
promedio_ingresos_pozo3 = np.mean(pozo3 * precio_petroleo)

print("Promedio de ingresos Pozo 1:", promedio_ingresos_pozo1, "USD")
print("Promedio de ingresos Pozo 2:", promedio_ingresos_pozo2, "USD")
print("Promedio de ingresos Pozo 3:", promedio_ingresos_pozo3, "USD")   






