# Archivo: lab_01_validacion.py
"""
Laboratorio 1: Validación Básica de Datos
Objetivo: Practicar el uso de operadores de comparación y condicionales
para validar datos operacionales.

La función debe validar:
1. La presión debe estar dentro de un rango seguro (min_presion a max_presion)
2. La temperatura debe estar por debajo de un valor máximo
3. El flujo debe ser mayor que cero

Si alguna validación falla, debe lanzar una excepción ValueError con 
el mensaje específico del error.
"""

def validar_datos_operacionales(presion, temperatura, flujo, min_presion, max_presion, max_temp):
    """
    Valida que los datos operacionales estén dentro de rangos seguros
    
    Args:
        presion (float): Presión actual del pozo en PSI
        temperatura (float): Temperatura actual en grados Celsius
        flujo (float): Flujo de producción en barriles por día
        min_presion (float): Presión mínima segura en PSI
        max_presion (float): Presión máxima segura en PSI
        max_temp (float): Temperatura máxima segura en grados Celsius
        
    Returns:
        bool: True si todos los datos son válidos
        
    Raises:
        ValueError: Cuando algún parámetro está fuera de rango
    """
    # TODO: Implementar las validaciones siguiendo estos pasos:
          
    # 1. Primero, validar que la presión esté dentro del rango seguro
    #    - Si está por debajo del mínimo, lanzar ValueError con mensaje 
    #      "Presión por debajo del mínimo seguro"
    #    - Si está por encima del máximo, lanzar ValueError con mensaje
    #      "Presión por encima del máximo seguro"

    # TODO: Escribe tu código de validación de presión aquí

    if presion < min_presion:
        raise ValueError("Presión por debajo del mínimo seguro")
    if presion > max_presion:
        raise ValueError("Presión por encima del máximo seguro")
    
        
    # 2. Luego, validar que la temperatura esté por debajo del máximo
    #    - Si está por encima del máximo, lanzar ValueError con mensaje
    #      "Temperatura excede el máximo seguro"
    
    # TODO: Escribe tu código de validación de temperatura aquí

    if temperatura > max_temp:
        raise ValueError("Temperatura excede el máximo seguro")
    
    
    # 3. Finalmente, validar que el flujo sea positivo
    #    - Si el flujo es cero o negativo, lanzar ValueError con mensaje
    #      "El flujo debe ser mayor que cero"
    
    # TODO: Escribe tu código de validación de flujo aquí

    if flujo <= 0:
        raise ValueError("El flujo debe ser mayor que cero")
    
    
    # 4. Si llegamos aquí, significa que todo está válido
    #    - Retornar True para indicar que los datos son válidos

    return True

    # Por ahora, lanzamos NotImplementedError para indicar que falta implementar
    raise NotImplementedError("¡Función no implementada! Debes escribir el código de validación.")
    

def main():
    # Casos de prueba
    casos_prueba = [
        # Caso válido (todos los parámetros dentro de rango)
        {"presion": 2200, "temperatura": 85, "flujo": 1500,
         "min_presion": 2000, "max_presion": 2500, "max_temp": 90},
        
        # Caso con presión por debajo del mínimo
        {"presion": 1900, "temperatura": 85, "flujo": 1500,
         "min_presion": 2000, "max_presion": 2500, "max_temp": 90},
        
        # Caso con presión por encima del máximo
        {"presion": 2600, "temperatura": 85, "flujo": 1500,
         "min_presion": 2000, "max_presion": 2500, "max_temp": 90},
        
        # Caso con temperatura excesiva
        {"presion": 2200, "temperatura": 95, "flujo": 1500,
         "min_presion": 2000, "max_presion": 2500, "max_temp": 90},
        
        # Caso con flujo negativo
        {"presion": 2200, "temperatura": 85, "flujo": -100,
         "min_presion": 2000, "max_presion": 2500, "max_temp": 90}
    ]
    
    print("=== Validación de Datos Operacionales ===")
    
    for i, caso in enumerate(casos_prueba, 1):
        print(f"\nCaso de prueba {i}:")
        print(f"Presión: {caso['presion']} PSI (Rango: {caso['min_presion']}-{caso['max_presion']} PSI)")
        print(f"Temperatura: {caso['temperatura']}°C (Máximo: {caso['max_temp']}°C)")
        print(f"Flujo: {caso['flujo']} bpd")
        
        try:
            validar_datos_operacionales(
                caso["presion"], caso["temperatura"], caso["flujo"],
                caso["min_presion"], caso["max_presion"], caso["max_temp"]
            )
            print("Resultado: ✅ Datos válidos")
        except NotImplementedError as e:
            print(f"Estado: {str(e)}")
        except ValueError as e:
            print(f"Resultado: ❌ Error - {str(e)}")

if __name__ == "__main__":
    main()