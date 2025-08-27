"""
Demo 02: Regresión para Predicción de Producción
================================================

Este demo muestra cómo implementar y evaluar modelos de regresión
para predecir la producción de petróleo usando variables operacionales.

Conceptos cubiertos:
- Regresión lineal simple y múltiple
- Random Forest para capturar no-linealidades
- Evaluación con métricas apropiadas
- Interpretación de resultados para el negocio
- Validación cruzada
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("DEMO 02: REGRESIÓN PARA PREDICCIÓN DE PRODUCCIÓN")
print("=" * 55)

# PASO 1: Preparar los datos (versión simplificada del demo anterior)
print("\n📊 PASO 1: PREPARACIÓN DE DATOS")
print("-" * 30)

# Cargar datos
df = pd.read_csv('/workspaces/CursoPython-Basico-JuanDavid/Sesión_16/datos/produccion_historica.csv')
print(f"Dataset cargado: {df.shape}")

# Preparación básica
df['fecha'] = pd.to_datetime(df['fecha'])
df = df.dropna(subset=['presion_boca_psi', 'temperatura_f', 'produccion_oil_bbl'])

# Features básicas para el modelo
features = ['presion_boca_psi', 'temperatura_f', 'dias_operacion', 'choke_size']
target = 'produccion_oil_bbl'

X = df[features]
y = df[target]

print(f"✅ Features preparadas: {X.shape}")
print(f"✅ Target preparado: {y.shape}")

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📊 División: {len(X_train)} train, {len(X_test)} test")

# PASO 2: Modelo de Regresión Lineal
print("\n🤖 PASO 2: MODELO DE REGRESIÓN LINEAL")
print("-" * 30)

# Entrenar modelo lineal
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train, y_train)

# Predicciones
y_pred_train_lineal = modelo_lineal.predict(X_train)
y_pred_test_lineal = modelo_lineal.predict(X_test)

# Métricas para regresión lineal
def calcular_metricas(y_true, y_pred, nombre):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n📈 Métricas - {nombre}:")
    print(f"   MAE (Error Absoluto Medio): {mae:.2f} bbl")
    print(f"   RMSE (Raíz del Error Cuadrático): {rmse:.2f} bbl")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAPE (Error Porcentual): {mape:.2f}%")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}

# Evaluar modelo lineal
metricas_lineal_train = calcular_metricas(y_train, y_pred_train_lineal, "Regresión Lineal - Train")
metricas_lineal_test = calcular_metricas(y_test, y_pred_test_lineal, "Regresión Lineal - Test")

# Mostrar coeficientes del modelo lineal
print(f"\n🔍 Coeficientes del modelo lineal:")
print(f"   Intercepto: {modelo_lineal.intercept_:.2f}")
for feature, coef in zip(features, modelo_lineal.coef_):
    print(f"   {feature}: {coef:.4f}")

# PASO 3: Modelo Random Forest
print("\n🌲 PASO 3: MODELO RANDOM FOREST")
print("-" * 30)

# Entrenar Random Forest
modelo_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
modelo_rf.fit(X_train, y_train)

# Predicciones
y_pred_train_rf = modelo_rf.predict(X_train)
y_pred_test_rf = modelo_rf.predict(X_test)

# Evaluar Random Forest
metricas_rf_train = calcular_metricas(y_train, y_pred_train_rf, "Random Forest - Train")
metricas_rf_test = calcular_metricas(y_test, y_pred_test_rf, "Random Forest - Test")

# Importancia de features en Random Forest
print(f"\n🔍 Importancia de features (Random Forest):")
importancias = modelo_rf.feature_importances_
indices_ordenados = np.argsort(importancias)[::-1]

for i in range(len(features)):
    idx = indices_ordenados[i]
    print(f"   {i+1}. {features[idx]}: {importancias[idx]:.4f}")

# PASO 4: Validación Cruzada
print("\n✅ PASO 4: VALIDACIÓN CRUZADA")
print("-" * 30)

# Validación cruzada para ambos modelos
cv_scores_lineal = cross_val_score(modelo_lineal, X_train, y_train, 
                                  cv=5, scoring='neg_mean_absolute_error')
cv_scores_rf = cross_val_score(modelo_rf, X_train, y_train, 
                              cv=5, scoring='neg_mean_absolute_error')

print(f"📊 Validación Cruzada (5-fold) - MAE:")
print(f"   Regresión Lineal: {-cv_scores_lineal.mean():.2f} ± {cv_scores_lineal.std():.2f}")
print(f"   Random Forest: {-cv_scores_rf.mean():.2f} ± {cv_scores_rf.std():.2f}")

# PASO 5: Visualización de Resultados
print("\n📊 PASO 5: VISUALIZACIÓN DE RESULTADOS")
print("-" * 30)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Predicción vs Real - Regresión Lineal
axes[0, 0].scatter(y_test, y_pred_test_lineal, alpha=0.6)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Producción Real (bbl)')
axes[0, 0].set_ylabel('Producción Predicha (bbl)')
axes[0, 0].set_title(f'Regresión Lineal\nR² = {metricas_lineal_test["r2"]:.3f}')

# Plot 2: Predicción vs Real - Random Forest
axes[0, 1].scatter(y_test, y_pred_test_rf, alpha=0.6, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Producción Real (bbl)')
axes[0, 1].set_ylabel('Producción Predicha (bbl)')
axes[0, 1].set_title(f'Random Forest\nR² = {metricas_rf_test["r2"]:.3f}')

# Plot 3: Residuales - Regresión Lineal
residuales_lineal = y_test - y_pred_test_lineal
axes[0, 2].scatter(y_pred_test_lineal, residuales_lineal, alpha=0.6)
axes[0, 2].axhline(y=0, color='r', linestyle='--')
axes[0, 2].set_xlabel('Predicción (bbl)')
axes[0, 2].set_ylabel('Residual (bbl)')
axes[0, 2].set_title('Residuales - Regresión Lineal')

# Plot 4: Residuales - Random Forest
residuales_rf = y_test - y_pred_test_rf
axes[1, 0].scatter(y_pred_test_rf, residuales_rf, alpha=0.6, color='green')
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_xlabel('Predicción (bbl)')
axes[1, 0].set_ylabel('Residual (bbl)')
axes[1, 0].set_title('Residuales - Random Forest')

# Plot 5: Importancia de Features
axes[1, 1].bar(range(len(features)), modelo_rf.feature_importances_)
axes[1, 1].set_xlabel('Feature')
axes[1, 1].set_ylabel('Importancia')
axes[1, 1].set_title('Importancia de Features (RF)')
axes[1, 1].set_xticks(range(len(features)))
axes[1, 1].set_xticklabels([f.replace('_', '\n') for f in features], rotation=45)

# Plot 6: Comparación de modelos
modelos = ['Regresión\nLineal', 'Random\nForest']
mae_scores = [metricas_lineal_test['mae'], metricas_rf_test['mae']]
r2_scores = [metricas_lineal_test['r2'], metricas_rf_test['r2']]

x_pos = np.arange(len(modelos))
axes[1, 2].bar(x_pos - 0.2, mae_scores, 0.4, label='MAE', alpha=0.7)
ax_twin = axes[1, 2].twinx()
ax_twin.bar(x_pos + 0.2, r2_scores, 0.4, label='R²', alpha=0.7, color='orange')
axes[1, 2].set_xlabel('Modelo')
axes[1, 2].set_ylabel('MAE (bbl)')
ax_twin.set_ylabel('R² Score')
axes[1, 2].set_title('Comparación de Modelos')
axes[1, 2].set_xticks(x_pos)
axes[1, 2].set_xticklabels(modelos)
axes[1, 2].legend(loc='upper left')
ax_twin.legend(loc='upper right')

plt.tight_layout()
plt.savefig('resultados_regresion.png', dpi=100, bbox_inches='tight')
print("📈 Gráficos guardados como 'resultados_regresion.png'")

# PASO 6: Análisis de Negocio
print("\n💼 PASO 6: ANÁLISIS DE IMPACTO EN NEGOCIO")
print("-" * 30)

# Cálculo de valor económico
precio_barril = 75  # USD por barril
dias_mes = 30

# Error promedio en predicción
error_promedio_lineal = metricas_lineal_test['mae']
error_promedio_rf = metricas_rf_test['mae']

# Impacto económico del error
impacto_economico_lineal = error_promedio_lineal * precio_barril * dias_mes
impacto_economico_rf = error_promedio_rf * precio_barril * dias_mes

print(f"💰 Impacto económico del error de predicción (mensual):")
print(f"   Regresión Lineal: ±${impacto_economico_lineal:,.0f}")
print(f"   Random Forest: ±${impacto_economico_rf:,.0f}")
print(f"   Ahorro potencial con RF: ${impacto_economico_lineal - impacto_economico_rf:,.0f}")

# Análisis de exactitud por rango de producción
print(f"\n📊 Exactitud por rango de producción:")
rangos = [(0, 800, 'Baja'), (800, 1200, 'Media'), (1200, float('inf'), 'Alta')]

for min_prod, max_prod, categoria in rangos:
    mask = (y_test >= min_prod) & (y_test < max_prod)
    if mask.sum() > 0:
        mae_rango_rf = mean_absolute_error(y_test[mask], y_pred_test_rf[mask])
        mape_rango_rf = np.mean(np.abs((y_test[mask] - y_pred_test_rf[mask]) / y_test[mask])) * 100
        print(f"   {categoria} producción: MAE = {mae_rango_rf:.1f} bbl, MAPE = {mape_rango_rf:.1f}%")

# PASO 7: Predicciones de ejemplo
print("\n🔮 PASO 7: EJEMPLOS DE PREDICCIÓN")
print("-" * 30)

# Casos de ejemplo para predicción
casos_ejemplo = [
    {'nombre': 'Operación Estándar', 'presion_boca_psi': 1500, 'temperatura_f': 180, 'dias_operacion': 200, 'choke_size': 32},
    {'nombre': 'Alta Presión', 'presion_boca_psi': 1800, 'temperatura_f': 180, 'dias_operacion': 200, 'choke_size': 32},
    {'nombre': 'Temperatura Elevada', 'presion_boca_psi': 1500, 'temperatura_f': 220, 'dias_operacion': 200, 'choke_size': 32},
    {'nombre': 'Choke Grande', 'presion_boca_psi': 1500, 'temperatura_f': 180, 'dias_operacion': 200, 'choke_size': 40},
    {'nombre': 'Pozo Envejecido', 'presion_boca_psi': 1200, 'temperatura_f': 170, 'dias_operacion': 800, 'choke_size': 28}
]

print(f"🎯 Predicciones con Random Forest:")
print(f"{'Escenario':<20} {'Predicción':<12} {'Confianza':<12}")
print("-" * 50)

for caso in casos_ejemplo:
    # Crear DataFrame con el caso
    X_caso = pd.DataFrame([caso])[features]
    
    # Predicción con Random Forest
    pred_rf = modelo_rf.predict(X_caso)[0]
    
    # Estimar intervalo de confianza usando predicciones de árboles individuales
    predicciones_arboles = [tree.predict(X_caso)[0] for tree in modelo_rf.estimators_]
    std_pred = np.std(predicciones_arboles)
    
    print(f"{caso['nombre']:<20} {pred_rf:.0f} bbl    ± {std_pred:.0f} bbl")

# PASO 8: Recomendaciones para optimización
print("\n🎯 PASO 8: RECOMENDACIONES DE OPTIMIZACIÓN")
print("-" * 30)

# Análisis de sensibilidad usando el modelo Random Forest
def analizar_sensibilidad(modelo, X_base, feature_idx, feature_name, valores_test):
    """Analiza cómo cambia la predicción al variar una feature"""
    predicciones = []
    X_test = X_base.copy()
    
    for valor in valores_test:
        X_test.iloc[0, feature_idx] = valor
        pred = modelo.predict(X_test)[0]
        predicciones.append(pred)
    
    return predicciones

# Caso base para análisis
caso_base = pd.DataFrame([{
    'presion_boca_psi': 1400,
    'temperatura_f': 180,
    'dias_operacion': 300,
    'choke_size': 32
}])

print("📈 Análisis de sensibilidad para optimización:")

# Sensibilidad de presión
presiones = np.arange(1200, 1801, 100)
pred_presion = analizar_sensibilidad(modelo_rf, caso_base, 0, 'presion_boca_psi', presiones)
mejor_presion_idx = np.argmax(pred_presion)
print(f"   Presión óptima: {presiones[mejor_presion_idx]} psi → {pred_presion[mejor_presion_idx]:.0f} bbl")

# Sensibilidad de choke
chokes = [24, 28, 32, 36, 40]
pred_choke = analizar_sensibilidad(modelo_rf, caso_base, 3, 'choke_size', chokes)
mejor_choke_idx = np.argmax(pred_choke)
print(f"   Choke óptimo: {chokes[mejor_choke_idx]}\" → {pred_choke[mejor_choke_idx]:.0f} bbl")

# RESUMEN FINAL
print("\n" + "="*55)
print("📋 RESUMEN DEL ANÁLISIS DE REGRESIÓN")
print("="*55)

print(f"🏆 Mejor modelo: Random Forest")
print(f"   MAE: {metricas_rf_test['mae']:.2f} bbl")
print(f"   R²: {metricas_rf_test['r2']:.4f}")
print(f"   MAPE: {metricas_rf_test['mape']:.2f}%")

print(f"\n💡 Insights clave:")
feature_importances_sorted = sorted(zip(features, modelo_rf.feature_importances_), 
                                   key=lambda x: x[1], reverse=True)
print(f"   • Variable más importante: {feature_importances_sorted[0][0]}")
print(f"   • El modelo explica {metricas_rf_test['r2']*100:.1f}% de la varianza")
print(f"   • Error promedio: {metricas_rf_test['mae']:.0f} bbl/día")

print(f"\n🎯 Próximos pasos recomendados:")
print(f"   1. Implementar monitoreo continuo del modelo")
print(f"   2. Reentrenar mensualmente con datos nuevos")
print(f"   3. Agregar más features (clima, mantenimiento)")
print(f"   4. Considerar modelos de ensemble más avanzados")

print(f"\n✅ Demo completado. Continúa con demo_03_clasificacion_eventos.py")