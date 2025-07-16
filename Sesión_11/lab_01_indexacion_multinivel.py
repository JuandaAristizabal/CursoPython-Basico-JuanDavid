# Archivo: lab_01_indexacion_multinivel.py
"""
Laboratorio 1: Indexación Multinivel y Jerarquías
Objetivo: Dominar las técnicas de indexación multinivel y operaciones jerárquicas
          con DataFrames para análisis avanzado de datos petroleros.

Este laboratorio utiliza datos del archivo pozos_multinivel.csv
"""
import pandas as pd
import numpy as np
from datetime import datetime

def main():
    # Cargar datos de ejemplo
    try:
        df_pozos = pd.read_csv('/workspaces/CursoPython-Basico-JuanDavid/Sesión_11/datos/pozos_multinivel.csv')
        print("Datos cargados exitosamente")
        print(f"Forma del DataFrame: {df_pozos.shape}")
        print(f"Columnas: {list(df_pozos.columns)}")
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'datos/pozos_multinivel.csv'")
        return
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return
    
    print("\n=== Laboratorio 1: Indexación Multinivel y Jerarquías ===\n")
    
    # EJERCICIO 1: Crear MultiIndex jerárquico
    # -------------------------------------------------------------------
    # Crea un MultiIndex usando las columnas 'campo', 'pozo' y 'fecha'
    # Luego establece este MultiIndex como índice del DataFrame

    print("Ejercicio 1: Crear MultiIndex jerárquico")
    
    # TO DO: Crea el MultiIndex aquí
    # Sugerencia: Usa pd.MultiIndex.from_arrays() o set_index()
    
    # Código de ejemplo para empezar:
    # df_multi = df_pozos.copy()
    # df_multi = df_multi.set_index(['campo', 'pozo', 'fecha'])

    # Crea un MultiIndex usando las columnas 'campo', 'pozo' y 'fecha'    
    df_multi = df_pozos.set_index(['campo', 'pozo', 'fecha'])

    # Luego establece este MultiIndex como índice del DataFrame
    df_multi.index.names = ['campo', 'pozo', 'fecha']

    # Imprimir resultados (no modificar)
    try:
        print(f"Forma del DataFrame con MultiIndex: {df_multi.shape}")
        print("Niveles del MultiIndex:")
        for i, level in enumerate(df_multi.index.names):
            print(f"  Nivel {i}: {level}")
        print("\nPrimeras filas del DataFrame con MultiIndex:")
        print(df_multi.head())
    except NameError:
        print("No se ha implementado el MultiIndex")
    
    print("\n" + "-"*50 + "\n")
    
    # EJERCICIO 2: Selección jerárquica avanzada
    # -------------------------------------------------------------------
    # Utiliza el DataFrame con MultiIndex para realizar selecciones jerárquicas:
    # a) Selecciona todos los datos del 'Campo Libertad'
    # b) Selecciona los datos del pozo 'LIB-001' en el 'Campo Libertad'
    # c) Selecciona los datos del 'Campo Libertad', pozo 'LIB-001' para '2024-01-01'
    
    print("Ejercicio 2: Selección jerárquica avanzada")
    
    # TO DO: Implementa las selecciones jerárquicas aquí
    
    # a) Selección por campo Libertad
    campo_libertad = df_multi.loc['Campo Libertad']
    
    # b) Selección por campo y pozo
    pozo_lib001 = df_multi.loc[('Campo Libertad', 'LIB-001')]
    
    # c) Selección por campo, pozo y fecha
    fecha_especifica = df_multi.loc[('Campo Libertad', 'LIB-001', '2024-01-01')]
    
    # Imprimir resultados (no modificar)
    try:
        print("a) Datos del Campo Libertad:")
        print(f"   Filas: {len(campo_libertad)}")
        print(f"   Pozos únicos: {campo_libertad.index.get_level_values('pozo').unique()}")
        
        print("\nb) Datos del pozo LIB-001 en Campo Libertad:")
        print(f"   Filas: {len(pozo_lib001)}")
        print(f"   Fechas: {list(pozo_lib001.index.get_level_values('fecha'))}")
        
        print("\nc) Datos específicos de fecha:")
        print(f"   Producción diaria: {fecha_especifica['produccion_diaria']} BPD")
        print(f"   Presión de cabeza: {fecha_especifica['presion_cabeza']} PSI")
    except NameError:
        print("No se han implementado las selecciones jerárquicas")
    
    print("\n" + "-"*50 + "\n")
    
    # EJERCICIO 3: Operaciones por niveles del MultiIndex
    # -------------------------------------------------------------------
    # Realiza operaciones agrupadas por diferentes niveles del MultiIndex:
    # a) Calcula la producción promedio por campo
    # b) Calcula la producción promedio por pozo dentro de cada campo
    # c) Encuentra el pozo con mayor producción promedio en cada campo
    
    print("Ejercicio 3: Operaciones por niveles del MultiIndex")
    
    # TO DO: Implementa las operaciones por niveles aquí
    
    # a) Producción promedio por campo
    produccion_por_campo = df_multi.groupby(level='campo')['produccion_diaria'].mean()
    
    # b) Producción promedio por pozo (manteniendo la jerarquía)
    produccion_por_pozo = df_multi.groupby(level=['campo', 'pozo'])['produccion_diaria'].mean()
    
    # c) Pozo con mayor producción por campo
    mejor_pozo_por_campo = df_multi.groupby(level='campo')['produccion_diaria'].agg(['mean', 'idxmax'])
    
    # Imprimir resultados (no modificar)
    try:
        print("a) Producción promedio por campo:")
        for campo, produccion in produccion_por_campo.items():
            print(f"   {campo}: {produccion:.2f} BPD")
        
        print("\nb) Producción promedio por pozo:")
        for (campo, pozo), produccion in produccion_por_pozo.items():
            print(f"   {campo} - {pozo}: {produccion:.2f} BPD")
        
        print("\nc) Mejor pozo por campo:")
        for campo in df_multi.index.get_level_values('campo').unique():
            mejor_idx = mejor_pozo_por_campo.loc[campo, 'idxmax']
            mejor_pozo = mejor_idx[1]  # El pozo está en el segundo nivel
            mejor_produccion = mejor_pozo_por_campo.loc[campo, 'mean']
            print(f"   {campo}: {mejor_pozo} ({mejor_produccion:.2f} BPD)")
    except NameError:
        print("No se han implementado las operaciones por niveles")
    
    print("\n" + "-"*50 + "\n")
    
    # EJERCICIO 4: Reestructuración de datos (Pivot y Stack/Unstack)
    # -------------------------------------------------------------------
    # Utiliza técnicas de reestructuración para crear vistas diferentes de los datos:
    # a) Crea un pivot table con campos como filas, pozos como columnas y producción como valores
    # b) Usa unstack() para convertir el nivel 'pozo' en columnas
    # c) Usa stack() para convertir columnas de nuevo a niveles del índice
    
    print("Ejercicio 4: Reestructuración de datos")
    
    # TO DO: Implementa las operaciones de reestructuración aquí
    
    # a) Pivot table
    df_pivot = df_pozos.pivot_table(
        index='campo', 
        columns='pozo', 
        values='produccion_diaria', 
        aggfunc='mean'
    )
    
    # b) Unstack del MultiIndex
    df_unstacked = df_multi['produccion_diaria'].unstack(level='pozo')
    
    # c) Stack de columnas
    df_stacked = df_unstacked.stack()
    
    # Imprimir resultados (no modificar)
    try:
        print("a) Pivot table (producción promedio por campo y pozo):")
        print(df_pivot)
        
        print("\nb) DataFrame unstacked (pozos como columnas):")
        print(df_unstacked)
        
        print("\nc) DataFrame re-stacked:")
        print(df_stacked.head(10))
    except NameError:
        print("No se han implementado las operaciones de reestructuración")
    
    print("\n" + "-"*50 + "\n")
    
    # EJERCICIO 5: Análisis comparativo avanzado
    # -------------------------------------------------------------------
    # Realiza un análisis comparativo sofisticado usando las técnicas aprendidas:
    # a) Calcula la eficiencia operacional por campo (producción/corte de agua)
    # b) Identifica los pozos con mayor variabilidad en producción
    # c) Crea un ranking de campos por rendimiento general
    
    print("Ejercicio 5: Análisis comparativo avanzado")
    
    # TO DO: Implementa el análisis comparativo aquí
    
    # a) Eficiencia operacional
    df_multi['eficiencia'] = df_multi['produccion_diaria'] / df_multi['agua_cut']
    eficiencia_por_campo = df_multi.groupby(level='campo')['eficiencia'].mean()
    
    # b) Variabilidad en producción
    variabilidad_pozos = df_multi.groupby(level=['campo', 'pozo'])['produccion_diaria'].agg(['mean', 'std'])
    variabilidad_pozos['coeficiente_variacion'] = variabilidad_pozos['std'] / variabilidad_pozos['mean']
    
    # c) Ranking de campos
    ranking_campos = df_multi.groupby(level='campo').agg({
        'produccion_diaria': ['mean', 'sum'],
        'gas_oil_ratio': 'mean',
        'agua_cut': 'mean'
    }).round(2)
    
    # Imprimir resultados (no modificar)
    try:
        print("a) Eficiencia operacional por campo:")
        for campo, eficiencia in eficiencia_por_campo.items():
            print(f"   {campo}: {eficiencia:.2f} BPD/agua_cut")
        
        print("\nb) Pozos con mayor variabilidad (CV > 0.1):")
        pozos_variables = variabilidad_pozos[variabilidad_pozos['coeficiente_variacion'] > 0.1]
        for (campo, pozo), datos in pozos_variables.iterrows():
            print(f"   {campo} - {pozo}: CV = {datos['coeficiente_variacion']:.3f}")
        
        print("\nc) Ranking de campos por rendimiento:")
        print(ranking_campos)
    except NameError:
        print("No se ha implementado el análisis comparativo")
    
    print("\n" + "-"*50 + "\n")
    
    # EJERCICIO 6: Operaciones condicionales con MultiIndex
    # -------------------------------------------------------------------
    # Aplica transformaciones condicionales usando el MultiIndex:
    # a) Marca los pozos con producción por encima del promedio de su campo
    # b) Calcula la desviación estandarizada de producción por campo
    # c) Identifica anomalías (valores fuera de 2 desviaciones estándar)
    
    print("Ejercicio 6: Operaciones condicionales con MultiIndex")
    
    # TO DO: Implementa las operaciones condicionales aquí
    
    # a) Marcado de pozos por encima del promedio
    df_multi['sobre_promedio'] = df_multi.groupby(level='campo')['produccion_diaria'].transform(
         lambda x: x > x.mean()
    )
    
    # b) Desviación estandarizada por campo
    df_multi['produccion_normalizada'] = df_multi.groupby(level='campo')['produccion_diaria'].transform(
         lambda x: (x - x.mean()) / x.std()
    )
    
    # c) Identificación de anomalías
    anomalias = df_multi[abs(df_multi['produccion_normalizada']) > 2]
    
    # Imprimir resultados (no modificar)
    try:
        print("a) Pozos con producción sobre el promedio de su campo:")
        pozos_sobre_promedio = df_multi[df_multi['sobre_promedio'] == True]
        for (campo, pozo, fecha), datos in pozos_sobre_promedio.iterrows():
            print(f"   {campo} - {pozo} ({fecha}): {datos['produccion_diaria']} BPD")
        
        print(f"\nb) Estadísticas de normalización:")
        print(f"   Media: {df_multi['produccion_normalizada'].mean():.3f}")
        print(f"   Desv. Est.: {df_multi['produccion_normalizada'].std():.3f}")
        
        print(f"\nc) Anomalías detectadas: {len(anomalias)} registros")
        for (campo, pozo, fecha), datos in anomalias.iterrows():
            print(f"   {campo} - {pozo} ({fecha}): {datos['produccion_diaria']} BPD (z-score: {datos['produccion_normalizada']:.2f})")
    except NameError:
        print("No se han implementado las operaciones condicionales")

if __name__ == "__main__":
    main() 