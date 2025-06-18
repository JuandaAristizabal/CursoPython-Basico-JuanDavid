def conversion_celsius_a_farenheit(celsius):
    """
    Convierte grados Celsius a Fahrenheit.
    
    Parámetros:
    celsius (float): Grados Celsius a convertir.
    
    Retorna:
    float: Grados Fahrenheit.
    """
    # Validar que el argumento debe ser un número
    if not isinstance(celsius, (int, float)):
        raise ValueError("El argumento debe ser un número.")
    
    return (celsius * 9/5) + 32

def conversion_farenheit_a_celsius(farenheit):
    """
    Convierte grados Fahrenheit a Celsius.
    
    Parámetros:
    farenheit (float): Grados Fahrenheit a convertir.
    
    Retorna:
    float: Grados Celsius.
    """
    # Validar que el argumento debe ser un número
    if not isinstance(farenheit, (int, float)):
        raise ValueError("El argumento debe ser un número.")
    
    return (farenheit - 32) * 5/9



