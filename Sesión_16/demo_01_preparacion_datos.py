"""
Demo 01: Preparación de Datos para Machine Learning
====================================================

Este demo muestra las técnicas fundamentales para preparar datos petroleros
antes de entrenar modelos de machine learning.

Conceptos cubiertos:
- Exploración inicial de datasets
- Limpieza y preprocesamiento
- Feature engineering específico para datos petroleros
- División de datos para entrenamiento/prueba
- Normalización y escalamiento
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("DEMO 01: PREPARACIÓN DE DATOS PARA ML")
print("=" * 50)

# PASO 1: Cargar y explorar los datos
print("\n🔍 PASO 1: EXPLORACIÓN INICIAL")
print("-" * 30)

# Cargar dataset de producción
df = pd.read_csv('/workspaces/CursoPython-Basico-JuanDavid/Sesión_16/datos/produccion_historica.csv')

print(f"📊 Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
print("\n📋 Primeras 5 filas:")
print(df.head())

print("\n📈 Información general:")
print(df.info())

print("\n🔢 Estadísticas descriptivas:")
print(df.describe())

# PASO 2: Análisis de calidad de datos
print("\n🔍 PASO 2: ANÁLISIS DE CALIDAD")
print("-" * 30)

# Valores faltantes
print("❌ Valores faltantes por columna:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
for col, count in missing.items():
    if count > 0:
        print(f"   {col}: {count} ({missing_pct[col]:.1f}%)")

# Duplicados
duplicates = df.duplicated().sum()
print(f"\n🔄 Filas duplicadas: {duplicates}")

# Valores anómalos básicos
print("\n⚠️  Detección de valores anómalos:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'dias_operacion':  # Excluir contador de días
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"   {col}: {outliers} outliers detectados")

# PASO 3: Limpieza de datos
print("\n🧹 PASO 3: LIMPIEZA DE DATOS")
print("-" * 30)

# Crear una copia para trabajar
df_clean = df.copy()

# Manejar valores faltantes
print("🔧 Aplicando estrategias de imputación:")

# Para presión: usar forward fill (valor anterior)
if df_clean['presion_boca_psi'].isnull().sum() > 0:
    before = df_clean['presion_boca_psi'].isnull().sum()
    df_clean['presion_boca_psi'].fillna(method='ffill', inplace=True)
    df_clean['presion_boca_psi'].fillna(df_clean['presion_boca_psi'].mean(), inplace=True)
    after = df_clean['presion_boca_psi'].isnull().sum()
    print(f"   ✓ presion_boca_psi: {before} → {after} valores faltantes")

# Para temperatura: usar interpolación lineal
if df_clean['temperatura_f'].isnull().sum() > 0:
    before = df_clean['temperatura_f'].isnull().sum()
    df_clean['temperatura_f'].interpolate(method='linear', inplace=True)
    df_clean['temperatura_f'].fillna(df_clean['temperatura_f'].mean(), inplace=True)
    after = df_clean['temperatura_f'].isnull().sum()
    print(f"   ✓ temperatura_f: {before} → {after} valores faltantes")

# PASO 4: Feature Engineering
print("\n⚙️ PASO 4: INGENIERÍA DE FEATURES")
print("-" * 30)

print("🔨 Creando nuevas features:")

# 1. Features temporales
df_clean['fecha'] = pd.to_datetime(df_clean['fecha'])
df_clean['dia_semana'] = df_clean['fecha'].dt.dayofweek
df_clean['mes'] = df_clean['fecha'].dt.month
df_clean['trimestre'] = df_clean['fecha'].dt.quarter
print("   ✓ Features temporales: dia_semana, mes, trimestre")

# 2. Ratios y relaciones
df_clean['ratio_gas_oil'] = df_clean['produccion_gas_mcf'] / df_clean['produccion_oil_bbl']
df_clean['corte_agua'] = df_clean['produccion_agua_bbl'] / (
    df_clean['produccion_oil_bbl'] + df_clean['produccion_agua_bbl']
) * 100
print("   ✓ Ratios: ratio_gas_oil, corte_agua")

# 3. Features de ventana móvil (últimos 7 días)
df_clean = df_clean.sort_values('fecha')
df_clean['produccion_promedio_7d'] = df_clean['produccion_oil_bbl'].rolling(window=7, min_periods=1).mean()
df_clean['produccion_std_7d'] = df_clean['produccion_oil_bbl'].rolling(window=7, min_periods=1).std()
df_clean['tendencia_7d'] = df_clean['produccion_oil_bbl'].rolling(window=7).apply(
    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 else 0
)
print("   ✓ Features de ventana: promedio_7d, std_7d, tendencia_7d")

# 4. Categorización de choke size
def categorizar_choke(size):
    if size <= 28:
        return 'pequeño'
    elif size <= 36:
        return 'mediano'
    else:
        return 'grande'

df_clean['choke_categoria'] = df_clean['choke_size'].apply(categorizar_choke)
print("   ✓ Feature categórica: choke_categoria")

# 5. Indicadores de performance
df_clean['performance_index'] = (
    df_clean['produccion_oil_bbl'] / 
    (df_clean['presion_boca_psi'] / 1000 * df_clean['temperatura_f'] / 100)
)
print("   ✓ Índice de performance creado")

# PASO 5: Visualización de features
print("\n📊 PASO 5: VISUALIZACIÓN DE FEATURES")
print("-" * 30)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

# Plot 1: Producción vs Presión
axes[0].scatter(df_clean['presion_boca_psi'], df_clean['produccion_oil_bbl'], alpha=0.6)
axes[0].set_xlabel('Presión Boca (psi)')
axes[0].set_ylabel('Producción Oil (bbl)')
axes[0].set_title('Relación Presión vs Producción')

# Plot 2: Producción vs Temperatura
axes[1].scatter(df_clean['temperatura_f'], df_clean['produccion_oil_bbl'], alpha=0.6, color='red')
axes[1].set_xlabel('Temperatura (°F)')
axes[1].set_ylabel('Producción Oil (bbl)')
axes[1].set_title('Relación Temperatura vs Producción')

# Plot 3: Producción por categoría de choke
df_clean.boxplot(column='produccion_oil_bbl', by='choke_categoria', ax=axes[2])
axes[2].set_xlabel('Categoría de Choke')
axes[2].set_ylabel('Producción Oil (bbl)')
axes[2].set_title('Producción por Categoría de Choke')

# Plot 4: Tendencia temporal
axes[3].plot(df_clean['fecha'].tail(100), df_clean['produccion_oil_bbl'].tail(100))
axes[3].set_xlabel('Fecha')
axes[3].set_ylabel('Producción Oil (bbl)')
axes[3].set_title('Tendencia Temporal (últimos 100 días)')
axes[3].tick_params(axis='x', rotation=45)

# Plot 5: Correlación entre features numéricas
numeric_features = ['presion_boca_psi', 'temperatura_f', 'produccion_oil_bbl', 
                   'ratio_gas_oil', 'corte_agua', 'performance_index']
correlation = df_clean[numeric_features].corr()
im = axes[4].imshow(correlation, cmap='coolwarm', aspect='auto')
axes[4].set_xticks(range(len(numeric_features)))
axes[4].set_yticks(range(len(numeric_features)))
axes[4].set_xticklabels(numeric_features, rotation=45)
axes[4].set_yticklabels(numeric_features)
axes[4].set_title('Matriz de Correlación')

# Plot 6: Distribución de la variable objetivo
axes[5].hist(df_clean['produccion_oil_bbl'], bins=30, edgecolor='black', alpha=0.7)
axes[5].set_xlabel('Producción Oil (bbl)')
axes[5].set_ylabel('Frecuencia')
axes[5].set_title('Distribución de Producción')

plt.tight_layout()
plt.savefig('exploracion_features.png', dpi=100, bbox_inches='tight')
print("📈 Visualizaciones guardadas como 'exploracion_features.png'")

# PASO 6: Preparar para ML
print("\n🤖 PASO 6: PREPARACIÓN FINAL PARA ML")
print("-" * 30)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Definir features y target
features_numericas = [
    'presion_boca_psi', 'temperatura_f', 'dias_operacion', 
    'choke_size', 'ratio_gas_oil', 'corte_agua',
    'produccion_promedio_7d', 'performance_index',
    'dia_semana', 'mes', 'trimestre'
]

features_categoricas = ['choke_categoria']
target = 'produccion_oil_bbl'

print(f"📝 Features seleccionadas:")
print(f"   • Numéricas: {len(features_numericas)}")
print(f"   • Categóricas: {len(features_categoricas)}")
print(f"   • Target: {target}")

# Preparar datos finales
df_final = df_clean.dropna(subset=features_numericas + [target])
print(f"\n✅ Dataset final: {len(df_final)} registros (se eliminaron {len(df_clean) - len(df_final)} por NaN)")

# Codificar variables categóricas
le = LabelEncoder()
for col in features_categoricas:
    df_final[f'{col}_encoded'] = le.fit_transform(df_final[col])

# Preparar matrices finales
X_num = df_final[features_numericas]
X_cat = df_final[[f'{col}_encoded' for col in features_categoricas]]
X = pd.concat([X_num, X_cat], axis=1)
y = df_final[target]

print(f"✅ Matriz de features X: {X.shape}")
print(f"✅ Vector target y: {y.shape}")

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

print(f"📊 División de datos:")
print(f"   • Entrenamiento: {len(X_train)} registros")
print(f"   • Prueba: {len(X_test)} registros")

# Escalamiento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("⚖️ Escalamiento aplicado con StandardScaler")

# Mostrar estadísticas del escalamiento
print(f"\nEstadísticas después del escalamiento:")
print(f"   • Media de features de entrenamiento: {X_train_scaled.mean(axis=0).mean():.3f}")
print(f"   • Std de features de entrenamiento: {X_train_scaled.std(axis=0).mean():.3f}")

# RESUMEN FINAL
print("\n" + "="*50)
print("📋 RESUMEN DEL PREPROCESSING")
print("="*50)

print(f"🔢 Dataset original: {df.shape[0]} registros")
print(f"🧹 Dataset limpio: {len(df_clean)} registros")
print(f"✅ Dataset final: {len(df_final)} registros")
print(f"📊 Features creadas: {len(X.columns)} total")
print(f"🎯 Listo para entrenamiento de modelos")

print("\n💡 Próximos pasos:")
print("   1. Entrenar modelo de regresión")
print("   2. Validar con datos de prueba")
print("   3. Evaluar métricas de performance")

print(f"\n✅ Demo completado. Continúa con demo_02_regresion_produccion.py")