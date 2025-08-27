#!/usr/bin/env python3
"""
Sesión 15: Integración de Datos de Múltiples Fuentes
Demo 02: Resolución de Conflictos entre Fuentes

Este demo demuestra técnicas para identificar y resolver inconsistencias
entre diferentes fuentes de datos en operaciones petroleras.

Autor: AdP Meridian Consulting
Fecha: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DEMO 02: RESOLUCIÓN DE CONFLICTOS ENTRE FUENTES")
print("="*70)
print()

# ===== SIMULACIÓN DE FUENTES CONFLICTIVAS =====
print("1. CREACIÓN DE FUENTES DE DATOS CON CONFLICTOS")
print("-"*50)

# Datos base de producción
np.random.seed(42)
fechas = pd.date_range('2024-01-01', '2024-01-07', freq='D')
pozos = ['PZ001', 'PZ002', 'PZ003']

# FUENTE 1: Sistema SCADA (Automático)
print("✓ Creando datos FUENTE 1 - Sistema SCADA:")
scada_data = []
for fecha in fechas:
    for pozo in pozos:
        # Simulamos lecturas automáticas con pequeños ruidos
        base_prod = {'PZ001': 150, 'PZ002': 90, 'PZ003': 240}[pozo]
        ruido = np.random.normal(0, 5)
        
        scada_data.append({
            'fecha': fecha,
            'pozo_id': pozo,
            'produccion_crudo_bpd': base_prod + ruido,
            'presion_cabeza_psi': 600 + np.random.normal(0, 20),
            'temperatura_c': 65 + np.random.normal(0, 3),
            'fuente': 'SCADA',
            'timestamp_registro': fecha + timedelta(hours=23, minutes=55),
            'calidad_dato': np.random.choice(['Excelente', 'Buena'], p=[0.8, 0.2])
        })

fuente_scada = pd.DataFrame(scada_data)
print(f"  - Registros: {len(fuente_scada)}")
print(f"  - Calidad promedio: {fuente_scada['calidad_dato'].value_counts().to_dict()}")
print()

# FUENTE 2: Reportes Operadores (Manual)
print("✓ Creando datos FUENTE 2 - Reportes Manuales:")
manual_data = []
for fecha in fechas:
    for pozo in pozos:
        # Simulamos reportes manuales con posibles errores humanos
        base_prod = {'PZ001': 150, 'PZ002': 90, 'PZ003': 240}[pozo]
        
        # Introducir errores típicos
        if np.random.random() < 0.1:  # 10% de error de transcripción
            error_factor = np.random.choice([0.1, 10])  # Decimal mal puesto
            produccion = base_prod * error_factor
        else:
            produccion = base_prod + np.random.normal(0, 10)
        
        # Algunos campos pueden estar vacíos
        presion = 600 + np.random.normal(0, 25) if np.random.random() > 0.05 else np.nan
        temperatura = 65 + np.random.normal(0, 5) if np.random.random() > 0.03 else np.nan
        
        manual_data.append({
            'fecha': fecha,
            'pozo_id': pozo,
            'produccion_crudo_bpd': produccion,
            'presion_cabeza_psi': presion,
            'temperatura_c': temperatura,
            'fuente': 'MANUAL',
            'timestamp_registro': fecha + timedelta(hours=np.random.randint(8, 18)),
            'operador': np.random.choice(['Juan P.', 'Maria R.', 'Carlos S.', 'Ana T.'])
        })

fuente_manual = pd.DataFrame(manual_data)
print(f"  - Registros: {len(fuente_manual)}")
print(f"  - Valores nulos en presión: {fuente_manual['presion_cabeza_psi'].isnull().sum()}")
print(f"  - Valores nulos en temperatura: {fuente_manual['temperatura_c'].isnull().sum()}")
print()

# FUENTE 3: Sistema de Medición Fiscal (Oficial)
print("✓ Creando datos FUENTE 3 - Medición Fiscal:")
fiscal_data = []
# Solo disponible cada 2 días y no para todos los pozos
for i, fecha in enumerate(fechas[::2]):  # Cada 2 días
    for pozo in pozos[:2]:  # Solo PZ001 y PZ002 tienen medición fiscal
        base_prod = {'PZ001': 150, 'PZ002': 90}[pozo]
        
        fiscal_data.append({
            'fecha': fecha,
            'pozo_id': pozo,
            'produccion_crudo_bpd': base_prod + np.random.normal(0, 2),  # Muy preciso
            'fuente': 'FISCAL',
            'timestamp_registro': fecha + timedelta(hours=12),
            'certificado': True,
            'precision': 'Alta'
        })

fuente_fiscal = pd.DataFrame(fiscal_data)
print(f"  - Registros: {len(fuente_fiscal)}")
print(f"  - Solo pozos: {sorted(fuente_fiscal['pozo_id'].unique())}")
print(f"  - Fechas disponibles: {len(fuente_fiscal['fecha'].unique())}")
print()

# ===== IDENTIFICACIÓN DE CONFLICTOS =====
print("2. IDENTIFICACIÓN DE CONFLICTOS ENTRE FUENTES")
print("-"*50)

# Combinar todas las fuentes para análisis
todas_fuentes = pd.concat([
    fuente_scada[['fecha', 'pozo_id', 'produccion_crudo_bpd', 'fuente']],
    fuente_manual[['fecha', 'pozo_id', 'produccion_crudo_bpd', 'fuente']],
    fuente_fiscal[['fecha', 'pozo_id', 'produccion_crudo_bpd', 'fuente']]
], ignore_index=True)

print("✓ Dataset consolidado de todas las fuentes:")
print(f"  - Total registros: {len(todas_fuentes)}")
print(f"  - Fuentes: {sorted(todas_fuentes['fuente'].unique())}")
print()

# Pivot para comparar fuentes lado a lado
pivot_fuentes = todas_fuentes.pivot_table(
    index=['fecha', 'pozo_id'],
    columns='fuente',
    values='produccion_crudo_bpd',
    aggfunc='first'
).reset_index()

print("✓ Comparación lado a lado de fuentes:")
print(pivot_fuentes.head(10))
print()

# Calcular diferencias entre fuentes
pivot_fuentes['diff_scada_manual'] = (
    pivot_fuentes['SCADA'] - pivot_fuentes['MANUAL']
).abs()

pivot_fuentes['diff_scada_fiscal'] = (
    pivot_fuentes['SCADA'] - pivot_fuentes['FISCAL']
).abs()

pivot_fuentes['diff_manual_fiscal'] = (
    pivot_fuentes['MANUAL'] - pivot_fuentes['FISCAL']
).abs()

# Identificar conflictos significativos (diferencia > 10%)
umbral_conflicto = 0.10  # 10%

def identificar_conflicto(row, col1, col2):
    if pd.isna(row[col1]) or pd.isna(row[col2]):
        return False
    diff_porcentual = abs(row[col1] - row[col2]) / max(row[col1], row[col2])
    return diff_porcentual > umbral_conflicto

pivot_fuentes['conflicto_scada_manual'] = pivot_fuentes.apply(
    lambda row: identificar_conflicto(row, 'SCADA', 'MANUAL'), axis=1
)

pivot_fuentes['conflicto_scada_fiscal'] = pivot_fuentes.apply(
    lambda row: identificar_conflicto(row, 'SCADA', 'FISCAL'), axis=1
)

print("✓ Análisis de conflictos:")
conflictos_sm = pivot_fuentes['conflicto_scada_manual'].sum()
conflictos_sf = pivot_fuentes['conflicto_scada_fiscal'].sum()
print(f"  - Conflictos SCADA vs MANUAL: {conflictos_sm}")
print(f"  - Conflictos SCADA vs FISCAL: {conflictos_sf}")

if conflictos_sm > 0:
    print("\n  Ejemplos de conflictos SCADA vs MANUAL:")
    conflictos_ejemplos = pivot_fuentes[pivot_fuentes['conflicto_scada_manual']].head(3)
    for _, row in conflictos_ejemplos.iterrows():
        pct_diff = abs(row['SCADA'] - row['MANUAL']) / max(row['SCADA'], row['MANUAL']) * 100
        print(f"    {row['fecha'].strftime('%Y-%m-%d')} {row['pozo_id']}: "
              f"SCADA={row['SCADA']:.1f}, MANUAL={row['MANUAL']:.1f} "
              f"(diff={pct_diff:.1f}%)")
print()

# ===== ESTRATEGIAS DE RESOLUCIÓN =====
print("3. ESTRATEGIAS DE RESOLUCIÓN DE CONFLICTOS")
print("-"*50)

# Estrategia 1: Jerarquía de fuentes
print("✓ ESTRATEGIA 1: Jerarquía de Fuentes")
jerarquia = {'FISCAL': 1, 'SCADA': 2, 'MANUAL': 3}  # 1 = máxima prioridad

def resolver_por_jerarquia(row):
    """Selecciona el valor de la fuente con mayor prioridad disponible."""
    for fuente in sorted(jerarquia.keys(), key=lambda x: jerarquia[x]):
        if not pd.isna(row[fuente]):
            return row[fuente], fuente
    return np.nan, 'NINGUNA'

pivot_fuentes[['valor_jerarquia', 'fuente_seleccionada']] = pivot_fuentes.apply(
    resolver_por_jerarquia, axis=1, result_type='expand'
)

print(f"  - Registros resueltos: {pivot_fuentes['fuente_seleccionada'].notna().sum()}")
fuente_counts = pivot_fuentes['fuente_seleccionada'].value_counts()
print(f"  - Distribución por fuente: {fuente_counts.to_dict()}")
print()

# Estrategia 2: Promedio ponderado
print("✓ ESTRATEGIA 2: Promedio Ponderado")
pesos = {'FISCAL': 0.6, 'SCADA': 0.3, 'MANUAL': 0.1}

def resolver_por_promedio_ponderado(row):
    """Calcula promedio ponderado de fuentes disponibles."""
    valores = []
    pesos_disponibles = []
    
    for fuente, peso in pesos.items():
        if not pd.isna(row[fuente]):
            valores.append(row[fuente])
            pesos_disponibles.append(peso)
    
    if valores:
        # Normalizar pesos
        suma_pesos = sum(pesos_disponibles)
        pesos_norm = [p/suma_pesos for p in pesos_disponibles]
        return sum(v*p for v, p in zip(valores, pesos_norm))
    return np.nan

pivot_fuentes['valor_promedio_ponderado'] = pivot_fuentes.apply(
    resolver_por_promedio_ponderado, axis=1
)

print(f"  - Registros con promedio calculado: {pivot_fuentes['valor_promedio_ponderado'].notna().sum()}")
print()

# Estrategia 3: Validación cruzada con umbrales
print("✓ ESTRATEGIA 3: Validación Cruzada")
def resolver_por_validacion_cruzada(row):
    """Usa consenso entre fuentes o rechaza outliers."""
    valores_disponibles = []
    fuentes_disponibles = []
    
    for fuente in ['FISCAL', 'SCADA', 'MANUAL']:
        if not pd.isna(row[fuente]):
            valores_disponibles.append(row[fuente])
            fuentes_disponibles.append(fuente)
    
    if len(valores_disponibles) < 2:
        return valores_disponibles[0] if valores_disponibles else np.nan, fuentes_disponibles[0] if fuentes_disponibles else 'NINGUNA'
    
    # Si tenemos múltiples valores, verificar consenso
    promedio = np.mean(valores_disponibles)
    valores_validos = []
    fuentes_validas = []
    
    # Umbral de desviación del 15%
    umbral = 0.15
    for valor, fuente in zip(valores_disponibles, fuentes_disponibles):
        desviacion = abs(valor - promedio) / promedio
        if desviacion <= umbral:
            valores_validos.append(valor)
            fuentes_validas.append(fuente)
    
    if valores_validos:
        # Usar promedio de valores válidos
        return np.mean(valores_validos), '+'.join(fuentes_validas)
    else:
        # Si no hay consenso, usar fuente más confiable
        return row['FISCAL'] if not pd.isna(row['FISCAL']) else row['SCADA'], 'FISCAL' if not pd.isna(row['FISCAL']) else 'SCADA'

pivot_fuentes[['valor_validacion_cruzada', 'fuentes_validacion']] = pivot_fuentes.apply(
    resolver_por_validacion_cruzada, axis=1, result_type='expand'
)

print(f"  - Registros validados: {pivot_fuentes['fuentes_validacion'].notna().sum()}")
validacion_counts = pivot_fuentes['fuentes_validacion'].value_counts()
print(f"  - Distribución de fuentes usadas: {validacion_counts.head().to_dict()}")
print()

# ===== COMPARACIÓN DE ESTRATEGIAS =====
print("4. COMPARACIÓN DE ESTRATEGIAS")
print("-"*50)

# Calcular métricas de comparación
estrategias = {
    'Jerarquía': 'valor_jerarquia',
    'Promedio Ponderado': 'valor_promedio_ponderado', 
    'Validación Cruzada': 'valor_validacion_cruzada'
}

print("✓ Estadísticas por estrategia:")
for nombre, columna in estrategias.items():
    datos_validos = pivot_fuentes[columna].dropna()
    print(f"\n  {nombre}:")
    print(f"    - Registros completos: {len(datos_validos)}")
    print(f"    - Promedio: {datos_validos.mean():.2f}")
    print(f"    - Desviación estándar: {datos_validos.std():.2f}")
    print(f"    - Rango: {datos_validos.min():.2f} - {datos_validos.max():.2f}")

# Calcular diferencias entre estrategias
print("\n✓ Diferencias entre estrategias:")
diff_j_p = (pivot_fuentes['valor_jerarquia'] - pivot_fuentes['valor_promedio_ponderado']).abs().mean()
diff_j_v = (pivot_fuentes['valor_jerarquia'] - pivot_fuentes['valor_validacion_cruzada']).abs().mean()
diff_p_v = (pivot_fuentes['valor_promedio_ponderado'] - pivot_fuentes['valor_validacion_cruzada']).abs().mean()

print(f"  - Jerarquía vs Promedio: {diff_j_p:.2f} bpd promedio")
print(f"  - Jerarquía vs Validación: {diff_j_v:.2f} bpd promedio")
print(f"  - Promedio vs Validación: {diff_p_v:.2f} bpd promedio")
print()

# ===== IMPLEMENTACIÓN DE REGLAS AUTOMÁTICAS =====
print("5. IMPLEMENTACIÓN DE REGLAS AUTOMÁTICAS")
print("-"*50)

def resolver_automatico(row):
    """
    Implementa lógica automática para resolver conflictos:
    1. Si existe FISCAL, usarlo (máxima confiabilidad)
    2. Si SCADA y MANUAL coinciden (±10%), usar promedio
    3. Si difieren significativamente, validar contra patrones históricos
    4. En caso de duda, usar SCADA (más frecuente y consistente)
    """
    
    # Regla 1: FISCAL tiene prioridad absoluta
    if not pd.isna(row['FISCAL']):
        return row['FISCAL'], 'FISCAL_DIRECTO'
    
    # Regla 2: Consenso SCADA-MANUAL
    if not pd.isna(row['SCADA']) and not pd.isna(row['MANUAL']):
        diff_pct = abs(row['SCADA'] - row['MANUAL']) / max(row['SCADA'], row['MANUAL'])
        if diff_pct <= 0.10:  # Diferencia menor al 10%
            return (row['SCADA'] + row['MANUAL']) / 2, 'CONSENSO_SM'
        else:
            # Validar contra rango esperado del pozo
            rangos_esperados = {
                'PZ001': (140, 160),
                'PZ002': (80, 100),
                'PZ003': (220, 250)
            }
            rango = rangos_esperados.get(row['pozo_id'], (0, 1000))
            
            scada_valido = rango[0] <= row['SCADA'] <= rango[1]
            manual_valido = rango[0] <= row['MANUAL'] <= rango[1]
            
            if scada_valido and not manual_valido:
                return row['SCADA'], 'SCADA_VALIDADO'
            elif manual_valido and not scada_valido:
                return row['MANUAL'], 'MANUAL_VALIDADO'
            else:
                # Ambos válidos o ambos inválidos: preferir SCADA
                return row['SCADA'], 'SCADA_PREFERENCIA'
    
    # Regla 3: Solo una fuente disponible
    if not pd.isna(row['SCADA']):
        return row['SCADA'], 'SCADA_UNICO'
    elif not pd.isna(row['MANUAL']):
        return row['MANUAL'], 'MANUAL_UNICO'
    
    return np.nan, 'SIN_DATOS'

pivot_fuentes[['valor_automatico', 'metodo_automatico']] = pivot_fuentes.apply(
    resolver_automatico, axis=1, result_type='expand'
)

print("✓ Resolución automática completada:")
metodo_counts = pivot_fuentes['metodo_automatico'].value_counts()
for metodo, count in metodo_counts.items():
    print(f"  - {metodo}: {count} casos")
print()

# ===== VISUALIZACIÓN DE CONFLICTOS Y RESOLUCIONES =====
print("6. VISUALIZACIÓN DE RESULTADOS")
print("-"*50)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Resolución de Conflictos entre Fuentes de Datos', fontsize=16, y=0.98)

# Gráfico 1: Comparación de fuentes para PZ001
pz001_data = pivot_fuentes[pivot_fuentes['pozo_id'] == 'PZ001'].copy()
ax1 = axes[0, 0]
x_pos = range(len(pz001_data))

if not pz001_data.empty:
    ax1.plot(x_pos, pz001_data['SCADA'], 'b-o', label='SCADA', linewidth=2)
    ax1.plot(x_pos, pz001_data['MANUAL'], 'r-s', label='MANUAL', linewidth=2)
    fiscal_data = pz001_data['FISCAL'].dropna()
    if not fiscal_data.empty:
        fiscal_x = [i for i, v in enumerate(pz001_data['FISCAL']) if not pd.isna(v)]
        ax1.plot(fiscal_x, fiscal_data, 'g-^', label='FISCAL', linewidth=3, markersize=8)
    
    ax1.set_xlabel('Día')
    ax1.set_ylabel('Producción (bpd)')
    ax1.set_title('PZ001 - Comparación de Fuentes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Marcar conflictos
    conflictos = pz001_data['conflicto_scada_manual']
    for i, conflicto in enumerate(conflictos):
        if conflicto:
            ax1.axvspan(i-0.3, i+0.3, alpha=0.3, color='yellow', label='Conflicto' if i == conflictos.idxmax() else "")

# Gráfico 2: Distribución de diferencias
ax2 = axes[0, 1]
diferencias = pivot_fuentes['diff_scada_manual'].dropna()
if not diferencias.empty:
    ax2.hist(diferencias, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(diferencias.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {diferencias.mean():.1f}')
    ax2.set_xlabel('Diferencia Absoluta (bpd)')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de Diferencias SCADA vs MANUAL')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# Gráfico 3: Comparación de estrategias
ax3 = axes[1, 0]
estrategias_data = []
for nombre, columna in estrategias.items():
    valores = pivot_fuentes[columna].dropna()
    if not valores.empty:
        estrategias_data.extend([(nombre, v) for v in valores])

if estrategias_data:
    estrategias_df = pd.DataFrame(estrategias_data, columns=['Estrategia', 'Valor'])
    sns.boxplot(data=estrategias_df, x='Estrategia', y='Valor', ax=ax3)
    ax3.set_title('Distribución de Valores por Estrategia')
    ax3.set_ylabel('Producción (bpd)')
    ax3.tick_params(axis='x', rotation=45)

# Gráfico 4: Métodos de resolución automática
ax4 = axes[1, 1]
metodo_counts = pivot_fuentes['metodo_automatico'].value_counts()
if not metodo_counts.empty:
    bars = ax4.bar(range(len(metodo_counts)), metodo_counts.values)
    ax4.set_xticks(range(len(metodo_counts)))
    ax4.set_xticklabels(metodo_counts.index, rotation=45, ha='right')
    ax4.set_ylabel('Número de Casos')
    ax4.set_title('Distribución de Métodos de Resolución')
    
    # Colorear barras según el tipo de método
    colors = ['green', 'blue', 'orange', 'red', 'purple', 'brown', 'gray']
    for bar, color in zip(bars, colors):
        bar.set_color(color)

plt.tight_layout()
plt.savefig('../datos/demo_02_conflictos.png', dpi=300, bbox_inches='tight')
print("✓ Gráficos guardados en: ../datos/demo_02_conflictos.png")
print()

# ===== EXPORTAR RESULTADOS =====
print("7. EXPORTACIÓN DE RESULTADOS")
print("-"*50)

# Dataset final con resolución automática
dataset_final = pivot_fuentes[['fecha', 'pozo_id', 'valor_automatico', 'metodo_automatico']].copy()
dataset_final.rename(columns={
    'valor_automatico': 'produccion_crudo_bpd',
    'metodo_automatico': 'metodo_resolucion'
}, inplace=True)

# Agregar métricas de calidad
dataset_final['numero_fuentes_disponibles'] = (
    pivot_fuentes[['SCADA', 'MANUAL', 'FISCAL']].notna().sum(axis=1)
)

dataset_final['confiabilidad'] = dataset_final['metodo_resolucion'].map({
    'FISCAL_DIRECTO': 'Muy Alta',
    'CONSENSO_SM': 'Alta',
    'SCADA_VALIDADO': 'Alta',
    'MANUAL_VALIDADO': 'Media',
    'SCADA_PREFERENCIA': 'Media',
    'SCADA_UNICO': 'Media',
    'MANUAL_UNICO': 'Baja',
    'SIN_DATOS': 'Sin Datos'
})

dataset_final.to_csv('../datos/produccion_conflictos_resueltos.csv', index=False)
print("✓ Dataset final guardado: ../datos/produccion_conflictos_resueltos.csv")

# Reporte de conflictos detectados
reporte_conflictos = pivot_fuentes[
    pivot_fuentes['conflicto_scada_manual'] | pivot_fuentes['conflicto_scada_fiscal']
].copy()

if not reporte_conflictos.empty:
    reporte_conflictos.to_csv('../datos/reporte_conflictos_detectados.csv', index=False)
    print("✓ Reporte de conflictos guardado: ../datos/reporte_conflictos_detectados.csv")
print()

# ===== RESUMEN EJECUTIVO =====
print("8. RESUMEN EJECUTIVO")
print("-"*50)
total_registros = len(pivot_fuentes)
registros_con_conflictos = (pivot_fuentes['conflicto_scada_manual'] | 
                           pivot_fuentes['conflicto_scada_fiscal']).sum()

print(f"✓ Resumen de la resolución de conflictos:")
print(f"  • Total de registros procesados: {total_registros}")
print(f"  • Registros con conflictos detectados: {registros_con_conflictos} ({registros_con_conflictos/total_registros*100:.1f}%)")
print(f"  • Registros resueltos automáticamente: {dataset_final['produccion_crudo_bpd'].notna().sum()}")
print()

print("✓ Distribución de confiabilidad final:")
conf_counts = dataset_final['confiabilidad'].value_counts()
for nivel, count in conf_counts.items():
    print(f"  • {nivel}: {count} registros ({count/len(dataset_final)*100:.1f}%)")
print()

print("✓ Técnicas implementadas:")
print("  • Jerarquía de fuentes basada en confiabilidad")
print("  • Promedio ponderado con pesos ajustables")
print("  • Validación cruzada con umbrales de consenso")
print("  • Validación contra rangos históricos esperados")
print("  • Resolución automática con reglas de negocio")
print()

print("="*70)
print("DEMO 02 COMPLETADO - RESOLUCIÓN DE CONFLICTOS")
print("="*70)

plt.show()