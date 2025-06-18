inventario = ["Crudo", "Gas Natural", "Gasolina"]
# Python discrimina entre mayúsculas y minúsculas en el ordenamiento de las listas (Mayusculas primero)
 
# Agregar un elemento al final de la lista
inventario.append ("Carbón")
inventario.append ("Gas Licuado")
inventario.append ("Diesel")
 
# Adicionar un elemento en la segunda posición de la lista
inventario.insert(1, "Hidrógeno")
print (inventario)
 
# Ordenar de la A a la Z
inventario.sort()
print (inventario)
 
# Ordenar de la Z a la A
inventario.sort(reverse=True)
print (inventario)
