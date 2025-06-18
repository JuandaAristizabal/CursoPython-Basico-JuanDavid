# Producciones diarias de la semana (lunes a domingo)
producciones = [145, 152, 148, 160, 155, 142, 158]
dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
 
print("=== ANÁLISIS DE PRODUCCIÓN SEMANAL ===")
 
# TODO 1: Calcula y muestra la producción total de la semana
# Tu código aquí:
produccion_total = sum(producciones)
print(f"Producción total de la semana: {produccion_total} unidades.")
 
# TODO 2: Encuentra el día con mayor producción (pista: usa max() e index())
# Tu código aquí:
max_produccion = max(producciones)
indice_max = producciones.index(max_produccion)
print(f"El día con mayor producción es {dias_semana[indice_max]} con {max_produccion} unidades.")
 
# TODO 3: Encuentra el día con menor producción
# Tu código aquí:
min_produccion = min(producciones)  
indice_min = producciones.index(min_produccion)
print(f"El día con menor producción es {dias_semana[indice_min]} con {min_produccion} unidades.")

# TODO 4: Calcula el promedio de producción diaria
# Tu código aquí:
promedio_produccion = produccion_total / len(producciones)
print(f"El promedio de producción diaria es {promedio_produccion:.2f} unidades.")
 
# TODO 5: Muestra las producciones de los primeros 3 días
# Tu código aquí:
primeros_3_dias = producciones[:3]
print(f"Producciones de los primeros 3 días:{primeros_3_dias}")
 
# TODO 6: Muestra las producciones de los primeros 3 días (otra forma)
# Tu código aquí:
print("Producciones de los primeros 3 días:")
for i in range(3):
    print(f"{dias_semana[i]}: {producciones[i]} unidades.")