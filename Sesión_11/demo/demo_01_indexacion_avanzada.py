# Archivo: demo_01_indexacion_avanzada.py
"""
Demostración 1: Indexación Avanzada y MultiIndex
Objetivo: Mostrar técnicas avanzadas de indexación y manipulación de datos jerárquicos
          aplicadas al sector petrolero.

Esta demostración complementa el Laboratorio 1
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def main():
    print("=== DEMOSTRACIÓN 1: INDEXACIÓN AVANZADA Y MULTIINDEX ===\n")
    
    # Cargar datos
    try:
        df_pozos = pd.read_csv('/workspaces/CursoPython-Basico-JuanDavid/Sesión_11/datos/pozos_multinivel.csv')
        print("✅ Datos cargados exitosamente")
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo de datos")
        return
    
    print(f"📊 Forma del DataFrame: {df_pozos.shape}")
    print(f"📋 Columnas: {list(df_pozos.columns)}")
    
    # ===================================================================
    # 1. CREACIÓN DE MULTIINDEX
    # ===================================================================
    print("\n" + "="*60)
    print("1. CREACIÓN DE MULTIINDEX")
    print("="*60)
    
    # Crear MultiIndex con campo, pozo y fecha
    df_multi = df_pozos.set_index(['campo', 'pozo', 'fecha'])
    
    print("🔧 MultiIndex creado con niveles: campo → pozo → fecha")
    print(f"📏 Forma: {df_multi.shape}")
    print(f"📏 Forma: {df_multi.columns}")
    print(f"🏗️  Niveles: {df_multi.index.names}")
    
    # Mostrar estructura del MultiIndex
    print("\n📋 Estructura del MultiIndex:")
    print(df_multi.head(10))
    
    # ===================================================================
    # 2. SELECCIÓN JERÁRQUICA
    # ===================================================================
    print("\n" + "="*60)
    print("2. SELECCIÓN JERÁRQUICA")
    print("="*60)
    
    # Selección por nivel
    print("🎯 Selección por campo completo:")
    campo_libertad = df_multi.loc['Campo Libertad']
    print(f"   Campo Libertad: {len(campo_libertad)} registros")
    print(f"   Pozos: {campo_libertad.index.get_level_values('pozo').unique()}")
    
    # Selección por múltiples niveles
    print("\n🎯 Selección por campo y pozo:")
    pozo_especifico = df_multi.loc[('Campo Libertad', 'LIB-001')]
    print(f"   LIB-001: {len(pozo_especifico)} registros")
    print(f"   Fechas: {list(pozo_especifico.index.get_level_values('fecha'))}")
    
    # Selección por todos los niveles
    print("\n🎯 Selección específica:")
    dato_especifico = df_multi.loc[('Campo Libertad', 'LIB-001', '2024-01-01')]
    print(f"   Producción: {dato_especifico['produccion_diaria']} BPD")
    print(f"   Presión: {dato_especifico['presion_cabeza']} PSI")
    
    # ===================================================================
    # 3. OPERACIONES POR NIVELES
    # ===================================================================
    print("\n" + "="*60)
    print("3. OPERACIONES POR NIVELES")
    print("="*60)
    
    # Agrupación por nivel de campo
    print("📊 Estadísticas por campo:")
    stats_por_campo = df_multi.groupby(level='campo').agg({
        'produccion_diaria': ['mean', 'sum', 'std'],
        'agua_cut': 'mean',
        'gas_oil_ratio': 'mean'
    }).round(2)
    print(stats_por_campo)
    
    # Agrupación por múltiples niveles
    print("\n📊 Estadísticas por campo y pozo:")
    stats_pozos = df_multi.groupby(level=['campo', 'pozo'])['produccion_diaria'].agg(['mean', 'std'])
    print(stats_pozos)
    
    # ===================================================================
    # 4. REESTRUCTURACIÓN DE DATOS
    # ===================================================================
    print("\n" + "="*60)
    print("4. REESTRUCTURACIÓN DE DATOS")
    print("="*60)
    
    # Pivot table
    print("🔄 Pivot table - Producción por campo y pozo:")
    df_pivot = df_pozos.pivot_table(
        index='campo',
        columns='pozo',
        values='produccion_diaria',
        aggfunc='mean'
    )
    print(df_pivot)
    
    # Unstack
    print("\n🔄 Unstack - Convertir nivel 'pozo' a columnas:")
    df_unstacked = df_multi['produccion_diaria'].unstack(level='pozo')
    print(df_unstacked)
    
    # Stack
    print("\n🔄 Stack - Convertir columnas de vuelta a niveles:")
    df_stacked = df_unstacked.stack()
    print(df_stacked.head(10))
    
    # ===================================================================
    # 5. TRANSFORMACIONES CONDICIONALES
    # ===================================================================
    print("\n" + "="*60)
    print("5. TRANSFORMACIONES CONDICIONALES")
    print("="*60)
    
    # Normalización por campo
    print("📈 Normalización de producción por campo:")
    df_multi['produccion_normalizada'] = df_multi.groupby(level='campo')['produccion_diaria'].transform(
        lambda x: (x - x.mean()) / x.std()
        #lambda x: (x - x.mean()) 
    )
    
    # Identificar valores atípicos
    anomalias = df_multi[abs(df_multi['produccion_normalizada']) > 2]
    print(f"🚨 Anomalías detectadas: {len(anomalias)} registros")
    
    if len(anomalias) > 0:
        print("   Ejemplos de anomalías:")
        for (campo, pozo, fecha), datos in anomalias.head(3).iterrows():
            print(f"   - {campo} {pozo} ({fecha}): {datos['produccion_diaria']} BPD (z-score: {datos['produccion_normalizada']:.2f})")
    
    # ===================================================================
    # 6. ANÁLISIS COMPARATIVO AVANZADO
    # ===================================================================
    print("\n" + "="*60)
    print("6. ANÁLISIS COMPARATIVO AVANZADO")
    print("="*60)
    
    # Eficiencia operacional
    df_multi['eficiencia_operacional'] = df_multi['produccion_diaria'] / (df_multi['presion_cabeza']/ 100)
    
    print("⚡ Eficiencia operacional por campo (BPD por 100 PSI):")
    eficiencia_por_campo = df_multi.groupby(level='campo')['eficiencia_operacional'].mean()
    for campo, eficiencia in eficiencia_por_campo.items():
        print(f"   {campo}: {eficiencia:.2f} BPD/100PSI")
    
    # Ranking de pozos
    print("\n🏆 Ranking de pozos por eficiencia:")
    ranking_pozos = df_multi.groupby(level=['campo', 'pozo'])['eficiencia_operacional'].mean().sort_values(ascending=False)
    for (campo, pozo), eficiencia in ranking_pozos.head(5).items():
        print(f"   {campo} - {pozo}: {eficiencia:.2f} BPD/100PSI")
    
    # ===================================================================
    # 7. APLICACIONES PRÁCTICAS
    # ===================================================================
    print("\n" + "="*60)
    print("7. APLICACIONES PRÁCTICAS")
    print("="*60)
    
    # Análisis de tendencias por campo
    print("📈 Análisis de tendencias por campo:")
    for campo in df_multi.index.get_level_values('campo').unique():
        datos_campo = df_multi.loc[campo]
        if len(datos_campo) > 1:
            # Calcular tendencia simple
            x = np.arange(len(datos_campo))
            y = datos_campo['produccion_diaria'].values
            tendencia = np.polyfit(x, y, 1)[0]
            
            direccion = "↗️ Aumentando" if tendencia > 0 else "↘️ Disminuyendo" if tendencia < 0 else "➡️ Estable"
            print(f"   {campo}: {direccion} (pendiente: {tendencia:.2f})")
    
    # Detección de patrones
    print("\n🔍 Detección de patrones:")
    
    # Pozos con mayor variabilidad
    variabilidad = df_multi.groupby(level=['campo', 'pozo'])['produccion_diaria'].agg(['mean', 'std'])
    variabilidad['cv'] = variabilidad['std'] / variabilidad['mean']
    
    pozos_variables = variabilidad[variabilidad['cv'] > 0.1]
    print(f"   Pozos con alta variabilidad (CV > 0.1): {len(pozos_variables)}")
    
    if len(pozos_variables) > 0:
        print("   Ejemplos:")
        for (campo, pozo), datos in pozos_variables.head(3).iterrows():
            print(f"   - {campo} {pozo}: CV = {datos['cv']:.3f}")
    
    # ===================================================================
    # 8. OPTIMIZACIONES Y MEJORES PRÁCTICAS
    # ===================================================================
    print("\n" + "="*60)
    print("8. OPTIMIZACIONES Y MEJORES PRÁCTICAS")
    print("="*60)
    
    print("💡 Consejos para trabajar con MultiIndex:")
    print("   • Usa .loc[] para selección jerárquica")
    print("   • Utiliza .groupby(level=...) para operaciones por niveles")
    print("   • Aplica .transform() para operaciones que mantienen la estructura")
    print("   • Considera .unstack() y .stack() para reestructuración")
    print("   • Usa .xs() para selección cruzada de niveles")
    
    print("\n⚡ Ventajas del MultiIndex:")
    print("   • Organización jerárquica natural de datos")
    print("   • Operaciones eficientes por grupos")
    print("   • Facilita análisis comparativos")
    print("   • Mejor rendimiento en operaciones complejas")
    
    print("\n🎯 Casos de uso en el sector petrolero:")
    print("   • Análisis por campo → pozo → fecha")
    print("   • Comparación de rendimiento entre activos")
    print("   • Detección de anomalías por contexto")
    print("   • Optimización de operaciones por zona")
    
    print("\n✅ Demostración completada exitosamente!")

if __name__ == "__main__":
    main() 