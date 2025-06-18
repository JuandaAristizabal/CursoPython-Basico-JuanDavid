# Archivo: demo_02_logicos.py
"""
Demo 2: Operadores Lógicos
Objetivo: Mostrar el uso de operadores lógicos (and, or, not) para combinar
condiciones en el contexto de validación de datos operacionales.
"""

def demo_logicos():
    print("=== Demo: Operadores Lógicos ===")
    
    # Ejemplo 1: Operador AND para validación de múltiples condiciones
    print("\nEjemplo 1: Validación con operador AND")
    
    # Datos del pozo
    presion = 2200      # PSI
    temperatura = 85    # °C
    flujo = 1500        # barriles por día
    
    # Rangos de operación segura
    presion_min, presion_max = 2000, 2500  # PSI
    temp_max = 90                          # °C
    flujo_min = 1000                       # barriles por día
    
    # Validación con AND: todas las condiciones deben ser verdaderas
    presion_ok = presion_min <= presion <= presion_max
    temperatura_ok = temperatura < temp_max
    flujo_ok = flujo >= flujo_min
    
    pozo_operando_seguro = presion_ok and temperatura_ok and flujo_ok
    
    print(f"Presión dentro de rango: {presion_ok}")
    print(f"Temperatura adecuada: {temperatura_ok}")
    print(f"Flujo suficiente: {flujo_ok}")
    print(f"¿El pozo está operando en condiciones seguras? {pozo_operando_seguro}")
    
    # Ejemplo 2: Operador OR para detección de alertas
    print("\nEjemplo 2: Detección de alertas con operador OR")
    
    # Nuevas lecturas (simulando problemas)
    presion_nueva = 2600      # PSI (por encima del máximo)
    temperatura_nueva = 86    # °C (aún bajo el máximo)
    flujo_nuevo = 800         # barriles por día (por debajo del mínimo)
    
    # Detección de condiciones de alerta (si cualquiera es verdadera)
    alerta_presion = presion_nueva < presion_min or presion_nueva > presion_max
    alerta_temperatura = temperatura_nueva >= temp_max
    alerta_flujo = flujo_nuevo < flujo_min
    
    hay_alertas = alerta_presion or alerta_temperatura or alerta_flujo
    
    print(f"Alerta de presión: {alerta_presion}")
    print(f"Alerta de temperatura: {alerta_temperatura}")
    print(f"Alerta de flujo: {alerta_flujo}")
    print(f"¿Se requiere atención del operador? {hay_alertas}")
    
    # Ejemplo 3: Operador NOT para invertir condiciones
    print("\nEjemplo 3: Uso del operador NOT")
    
    mantenimiento_programado = False
    pozo_activo = True
    
    # Determinar si el pozo debería estar produciendo
    debe_producir = pozo_activo and not mantenimiento_programado
    
    print(f"Pozo activo: {pozo_activo}")
    print(f"Mantenimiento programado: {mantenimiento_programado}")
    print(f"¿El pozo debe estar produciendo? {debe_producir}")
    
    # Ejemplo 4: Combinación de operadores para condiciones complejas
    print("\nEjemplo 4: Condiciones complejas")
    
    # Parámetros adicionales
    agua_en_produccion = 15  # porcentaje
    gas_en_produccion = 5    # porcentaje
    
    # Condiciones límite
    agua_max = 20
    gas_max = 10
    
    # Condición compleja: operación óptima
    operacion_optima = (
        (presion_min <= presion <= presion_max) and  # presión en rango
        (temperatura < temp_max) and                  # temperatura adecuada
        (flujo >= flujo_min) and                      # flujo suficiente
        (agua_en_produccion < agua_max) and           # agua bajo control
        (gas_en_produccion < gas_max)                 # gas bajo control
    )
    
    print(f"Agua en producción: {agua_en_produccion}% (máx: {agua_max}%)")
    print(f"Gas en producción: {gas_en_produccion}% (máx: {gas_max}%)")
    print(f"¿El pozo está operando en condiciones óptimas? {operacion_optima}")

if __name__ == "__main__":
    demo_logicos()
