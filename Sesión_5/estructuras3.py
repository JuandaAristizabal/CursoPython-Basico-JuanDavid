# Lista inicial del inventario
inventario = ["Crudo", "Gas Natural", "Gasolina"]
 
print("=== GESTIÓN DE INVENTARIO PETROLERO ===")
print(f"Inventario inicial: {inventario}")
 
# TODO 1: Agrega "Diesel" al final de la lista
# Tu código aquí:
inventario.append("Diesel")
print(f"Inventario después de agregar Diesel: {inventario}")
 
# TODO 2: Inserta "GLP" (Gas Licuado de Petróleo) en la segunda posición (índice 1)
# Tu código aquí:
inventario.insert(1, "GLP")
print(f"Inventario después de insertar GLP: {inventario}")
 
# TODO 3: Elimina el tercer elemento de la lista actual
# Tu código aquí:
inventario.pop(2)
print(f"Inventario después de eliminar el tercer elemento: {inventario}")
 
# TODO 4: Ordena la lista alfabéticamente
# Tu código aquí:
inventario.sort()
 
 
print(f"Inventario final: {inventario}")
