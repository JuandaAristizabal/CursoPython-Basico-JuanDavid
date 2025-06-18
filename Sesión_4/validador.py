# Función que valida si la presión se encuentra en un rango seguro
def validar_presion(presion):
    """
    Valida si la presión se encuentra en un rango seguro.
    
    Parámetros:
    presion (float): Presión a validar.
    
    Retorna:
    bool: True si la presión es segura, False de lo contrario.
    """
    # Validar que el argumento debe ser un número
    if not isinstance(presion, (int, float)):
        return ValueError("El argumento debe ser un número.")
    
    if presion < 100:
        return "La presión se encuentra en rango seguro."
    else:
        return "La presión se encuentra en rango peligroso."
    
 # Calcular el volumen total de producción y devolver el resultado
def calcular_volumen_total_produccion(produccion):
    volumen_total = 0

    for presion_individual in produccion:
        # Validar que el argumento debe ser un número
        if not isinstance(presion_individual, (int, float)):
            return ValueError("El argumento debe ser un número.")
        
        # Validar que la presión se encuentra en rango seguro
        if validar_presion(presion_individual) == "La presión se encuentra en rango seguro.":
            volumen_total = volumen_total + presion_individual
            #print(f"Volumen total de producción: {volumen_total}")

    return volumen_total

