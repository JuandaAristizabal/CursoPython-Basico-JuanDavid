# Archivo: lab_02_funciones_avanzadas.py
"""
Laboratorio 2: Funciones Avanzadas y Vectorización
Objetivo: Dominar las funciones apply, map, applymap y técnicas de vectorización
          para transformaciones complejas de datos petroleros.

Este laboratorio utiliza datos del archivo sensores_avanzado.csv
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def main():
    # Cargar datos de ejemplo
    try:
        df_sensores = pd.read_csv('/workspaces/CursoPython-Basico-JuanDavid/Sesión_11/datos/sensores_avanzado.csv')
        print("Datos cargados exitosamente")
        print(f"Forma del DataFrame: {df_sensores.shape}")
        print(f"Columnas: {list(df_sensores.columns)}")
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'datos/sensores_avanzado.csv'")
        return
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return
    
    print("\n=== Laboratorio 2: Funciones Avanzadas y Vectorización ===\n")
    
    # EJERCICIO 1: Función apply() con Series
    # -------------------------------------------------------------------
    # Utiliza apply() para transformar datos de sensores:
    # a) Convierte la columna 'fecha_hora' a datetime
    # b) Crea una función que calcule el estado de alarma basado en límites
    # c) Aplica la función a cada fila para determinar si hay alarma
    
    print("Ejercicio 1: Función apply() con Series")
    
    # TODO: Implementa las transformaciones con apply() aquí
   
    # Imprimir resultados (no modificar)
    try:
        print("a) Tipos de datos después de conversión:")
        print(f"   fecha_hora: {df_sensores['fecha_hora'].dtype}")
        print(f"   Rango de fechas: {df_sensores['fecha_hora'].min()} a {df_sensores['fecha_hora'].max()}")
        
        print("\nb) Distribución de estados de alarma:")
        print(df_sensores['estado_alarma'].value_counts())
        
        print("\nc) Ejemplos de alarmas:")
        alarmas = df_sensores[df_sensores['estado_alarma'] == 'ALARMA']
        for _, row in alarmas.head(3).iterrows():
            print(f"   {row['sensor_id']}: {row['valor']} {row['unidad']} (límites: {row['limite_min']}-{row['limite_max']})")
    except NameError:
        print("No se han implementado las transformaciones con apply()")
    
    print("\n" + "-"*50 + "\n")
    
    # EJERCICIO 2: Función map() para transformaciones de valores
    # -------------------------------------------------------------------
    # Utiliza map() para transformaciones de valores discretos:
    # a) Crea un mapeo de prioridades a valores numéricos
    # b) Mapea tipos de sensores a categorías de criticidad
    # c) Crea un mapeo de ubicaciones a zonas operacionales
    
    print("Ejercicio 2: Función map() para transformaciones de valores")
    
    # TODO: Implementa las transformaciones con map() aquí
    
    # Imprimir resultados (no modificar)
    try:
        print("a) Distribución de prioridades numéricas:")
        print(df_sensores['prioridad_numerica'].value_counts().sort_index())
        
        print("\nb) Distribución por criticidad:")
        print(df_sensores['criticidad'].value_counts())
        
        print("\nc) Distribución por zona operacional:")
        print(df_sensores['zona_operacional'].value_counts())
        
        print("\nd) Sensores críticos por zona:")
        sensores_criticos = df_sensores[df_sensores['criticidad'] == 'Crítico']
        print(sensores_criticos.groupby('zona_operacional')['sensor_id'].count())
    except NameError:
        print("No se han implementado las transformaciones con map()")
    
    print("\n" + "-"*50 + "\n")
    
    # EJERCICIO 3: Función applymap() para transformaciones de DataFrame
    # -------------------------------------------------------------------
    # Utiliza applymap() para transformaciones elementales:
    # a) Crea una función que normalice valores numéricos
    # b) Aplica la función a columnas numéricas específicas
    # c) Crea una función de formateo para valores
    
    print("Ejercicio 3: Función applymap() para transformaciones de DataFrame")
    
    # TODO: Implementa las transformaciones con applymap() aquí
    
    # Imprimir resultados (no modificar)
    try:
        print("a) Valores originales vs normalizados:")
        print("   Originales:")
        print(df_sensores[columnas_numericas].head())
        print("\n   Normalizados:")
        print(df_normalizado.head())
        
        print("\nb) Estadísticas de valores normalizados:")
        print(df_normalizado.describe())
        
        print("\nc) Ejemplo de formateo:")
        valores_formateados = df_sensores[columnas_numericas].applymap(formatear_valor)
        print(valores_formateados.head())
    except NameError:
        print("No se han implementado las transformaciones con applymap()")
    
    print("\n" + "-"*50 + "\n")
    
    # EJERCICIO 4: Vectorización con NumPy y Pandas
    # -------------------------------------------------------------------
    # Implementa operaciones vectorizadas para mejor rendimiento:
    # a) Calcula la desviación de valores respecto a límites usando vectorización
    # b) Crea indicadores de rendimiento usando operaciones vectorizadas
    # c) Calcula métricas de eficiencia operacional
    
    print("Ejercicio 4: Vectorización con NumPy y Pandas")
    
    # TODO: Implementa las operaciones vectorizadas aquí
    
    
    # Imprimir resultados (no modificar)
    try:
        print("a) Estadísticas de desviación:")
        print(f"   Desviación mínima promedio: {df_sensores['desviacion_min'].mean():.3f}")
        print(f"   Desviación máxima promedio: {df_sensores['desviacion_max'].mean():.3f}")
        
        print("\nb) Distribución de porcentaje en rango:")
        print(df_sensores['porcentaje_rango'].describe())
        
        print("\nc) Eficiencia operacional por tipo de sensor:")
        eficiencia_por_tipo = df_sensores.groupby('tipo_sensor')['eficiencia_operacional'].mean()
        for tipo, eficiencia in eficiencia_por_tipo.items():
            print(f"   {tipo}: {eficiencia:.3f}")
        
        print(f"\nd) Sensores activos: {df_sensores['indicador_estado'].sum()} de {len(df_sensores)}")
    except NameError:
        print("No se han implementado las operaciones vectorizadas")
    
    print("\n" + "-"*50 + "\n")
    
    # EJERCICIO 5: Funciones personalizadas complejas
    # -------------------------------------------------------------------
    # Crea funciones personalizadas que combinen múltiples criterios:
    # a) Función que calcule un score de salud del sensor
    # b) Función que determine la urgencia de mantenimiento
    # c) Función que calcule la confiabilidad del sensor
    
    print("Ejercicio 5: Funciones personalizadas complejas")
    
    # TODO: Implementa las funciones personalizadas aquí
    
    # Imprimir resultados (no modificar)
    try:
        print("a) Distribución de salud de sensores:")
        print(df_sensores['salud_sensor'].describe())
        
        print("\nb) Sensores con urgencia alta (4-5):")
        sensores_urgentes = df_sensores[df_sensores['urgencia_mantenimiento'] >= 4]
        for _, row in sensores_urgentes.iterrows():
            print(f"   {row['sensor_id']}: Urgencia {row['urgencia_mantenimiento']}, Salud {row['salud_sensor']}")
        
        print("\nc) Confiabilidad promedio por criticidad:")
        confiabilidad_por_criticidad = df_sensores.groupby('criticidad')['confiabilidad'].mean()
        for criticidad, conf in confiabilidad_por_criticidad.items():
            print(f"   {criticidad}: {conf:.3f}")
        
        print("\nd) Sensores con baja salud (< 50):")
        sensores_baja_salud = df_sensores[df_sensores['salud_sensor'] < 50]
        print(f"   Total: {len(sensores_baja_salud)} sensores")
        for _, row in sensores_baja_salud.head(3).iterrows():
            print(f"   {row['sensor_id']}: Salud {row['salud_sensor']}, Urgencia {row['urgencia_mantenimiento']}")
    except NameError:
        print("No se han implementado las funciones personalizadas")
    
    print("\n" + "-"*50 + "\n")
    
    # EJERCICIO 6: Optimización de rendimiento
    # -------------------------------------------------------------------
    # Compara el rendimiento de diferentes enfoques:
    # a) Mide el tiempo de apply() vs vectorización
    # b) Optimiza una operación compleja
    # c) Crea una función eficiente para procesamiento en lotes
    
    print("Ejercicio 6: Optimización de rendimiento")
    
    # TODO: Implementa la comparación de rendimiento aquí
    
    import time
    
    # Imprimir resultados (no modificar)
    try:
        print("a) Comparación de rendimiento:")
        print(f"   Tiempo con apply(): {tiempo_apply:.4f} segundos")
        print(f"   Tiempo con vectorización: {tiempo_vector:.4f} segundos")
        print(f"   Mejora de rendimiento: {tiempo_apply/tiempo_vector:.1f}x más rápido")
        
        print("\nb) Verificación de resultados:")
        print(f"   Resultados iguales: {resultado_apply.equals(resultado_vector)}")
        
        print("\nc) Procesamiento en lotes optimizado:")
        tiempo_lotes = time.time()
        resultado_lotes = procesar_lote_optimizado(df_sensores)
        tiempo_lotes = time.time() - tiempo_lotes
        print(f"   Tiempo con lotes: {tiempo_lotes:.4f} segundos")
        print(f"   Tamaño del resultado: {len(resultado_lotes)}")
    except NameError:
        print("No se ha implementado la comparación de rendimiento")

if __name__ == "__main__":
    main() 