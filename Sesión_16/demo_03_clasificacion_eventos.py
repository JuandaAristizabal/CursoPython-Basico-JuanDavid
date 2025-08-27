"""
Demo 03: Clasificación de Eventos Operacionales
===============================================

Este demo muestra cómo implementar modelos de clasificación para
detectar y categorizar eventos operacionales en la industria petrolera.

Conceptos cubiertos:
- Clasificación multiclase de tipos de eventos
- Manejo de datasets desbalanceados
- Métricas de evaluación para clasificación
- Análisis de costos de clasificación errónea
- Interpretabilidad de modelos de árboles de decisión
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_recall_fscore_support)
import warnings
warnings.filterwarnings('ignore')

print("DEMO 03: CLASIFICACIÓN DE EVENTOS OPERACIONALES")
print("=" * 55)

# PASO 1: Cargar y explorar datos de eventos
print("\n📊 PASO 1: EXPLORACIÓN DE DATOS DE EVENTOS")
print("-" * 30)

# Cargar dataset de eventos
df_eventos = pd.read_csv('/workspaces/CursoPython-Basico-JuanDavid/Sesión_16/datos/eventos_operacionales.csv')
print(f"Dataset cargado: {df_eventos.shape}")

# Explorar tipos de eventos
print(f"\n📋 Distribución de tipos de eventos:")
distribucion = df_eventos['tipo_evento'].value_counts()
print(distribucion)
print(f"\n📊 Porcentaje por tipo:")
print((distribucion / len(df_eventos) * 100).round(1))

# Cargar datos de sensores para crear features predictivas
df_sensores = pd.read_csv('/workspaces/CursoPython-Basico-JuanDavid/Sesión_16/datos/anomalias_sensores.csv')
print(f"\nDatos de sensores: {df_sensores.shape}")

# Simular integración de datos de sensores con eventos
# (en la realidad harías merge por timestamp/sensor_id)
np.random.seed(42)
n_eventos = len(df_eventos)

# Crear features sintéticas que representen el estado de los sensores
# antes del evento
df_eventos['temperatura_sensor'] = 180 + np.random.normal(0, 15, n_eventos)
df_eventos['vibracion_sensor'] = 2.5 + np.random.normal(0, 0.8, n_eventos)
df_eventos['presion_sensor'] = 1500 + np.random.normal(0, 120, n_eventos)
df_eventos['ruido_sensor'] = 85 + np.random.normal(0, 8, n_eventos)

# Agregar variación según el tipo de evento
for i, evento in enumerate(df_eventos['tipo_evento']):
    if evento == 'falla_bomba':
        df_eventos.loc[i, 'vibracion_sensor'] *= np.random.uniform(2, 4)
        df_eventos.loc[i, 'ruido_sensor'] += np.random.uniform(15, 30)
    elif evento == 'obstruccion':
        df_eventos.loc[i, 'presion_sensor'] *= np.random.uniform(0.6, 0.8)
        df_eventos.loc[i, 'temperatura_sensor'] += np.random.uniform(20, 40)
    elif evento == 'fuga_menor':
        df_eventos.loc[i, 'presion_sensor'] *= np.random.uniform(0.8, 0.9)

print(f"✅ Features de sensores creadas")

# PASO 2: Preparación de datos para clasificación
print("\n⚙️ PASO 2: PREPARACIÓN DE DATOS")
print("-" * 30)

# Crear features adicionales
df_eventos['ratio_temp_presion'] = df_eventos['temperatura_sensor'] / df_eventos['presion_sensor']
df_eventos['indice_anomalia'] = (
    (df_eventos['temperatura_sensor'] - 180) / 15 +
    (df_eventos['vibracion_sensor'] - 2.5) / 0.8 +
    (df_eventos['ruido_sensor'] - 85) / 8
) / 3

# Agregar features temporales
df_eventos['fecha'] = pd.to_datetime(df_eventos['fecha'])
df_eventos['hora'] = df_eventos['fecha'].dt.hour
df_eventos['dia_semana'] = df_eventos['fecha'].dt.dayofweek
df_eventos['es_fin_semana'] = (df_eventos['dia_semana'] >= 5).astype(int)

print(f"✅ Features adicionales creadas")

# Definir features y target
features = [
    'temperatura_sensor', 'vibracion_sensor', 'presion_sensor', 'ruido_sensor',
    'duracion_horas', 'ratio_temp_presion', 'indice_anomalia',
    'hora', 'dia_semana', 'es_fin_semana'
]

# Filtrar solo eventos que requieren clasificación (excluir 'normal')
df_clasificacion = df_eventos[df_eventos['tipo_evento'] != 'normal'].copy()
print(f"📊 Eventos para clasificar: {len(df_clasificacion)} (excluyendo 'normal')")

# Preparar matrices
X = df_clasificacion[features]
y = df_clasificacion['tipo_evento']

# Codificar labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
clases = le.classes_
print(f"🏷️  Clases a predecir: {list(clases)}")

# División train/test con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"📊 División: {len(X_train)} train, {len(X_test)} test")

# PASO 3: Modelo de Árbol de Decisión
print("\n🌳 PASO 3: MODELO DE ÁRBOL DE DECISIÓN")
print("-" * 30)

# Entrenar árbol de decisión
modelo_arbol = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    class_weight='balanced'  # Para manejar desbalance
)

modelo_arbol.fit(X_train, y_train)

# Predicciones
y_pred_arbol = modelo_arbol.predict(X_test)
y_pred_proba_arbol = modelo_arbol.predict_proba(X_test)

# Métricas
accuracy_arbol = accuracy_score(y_test, y_pred_arbol)
print(f"🎯 Precisión del árbol: {accuracy_arbol:.3f}")

# Reporte de clasificación
print(f"\n📊 Reporte de clasificación (Árbol de Decisión):")
print(classification_report(y_test, y_pred_arbol, target_names=clases, digits=3))

# Matriz de confusión
matriz_confusion_arbol = confusion_matrix(y_test, y_pred_arbol)
print(f"\n📊 Matriz de confusión:")
print(matriz_confusion_arbol)

# PASO 4: Modelo Random Forest
print("\n🌲 PASO 4: MODELO RANDOM FOREST")
print("-" * 30)

# Entrenar Random Forest
modelo_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

modelo_rf.fit(X_train, y_train)

# Predicciones
y_pred_rf = modelo_rf.predict(X_test)
y_pred_proba_rf = modelo_rf.predict_proba(X_test)

# Métricas
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"🎯 Precisión del Random Forest: {accuracy_rf:.3f}")

print(f"\n📊 Reporte de clasificación (Random Forest):")
print(classification_report(y_test, y_pred_rf, target_names=clases, digits=3))

# Importancia de features
print(f"\n🔍 Importancia de features (Random Forest):")
importancias = modelo_rf.feature_importances_
indices_ord = np.argsort(importancias)[::-1]

for i in range(len(features)):
    idx = indices_ord[i]
    print(f"   {i+1}. {features[idx]}: {importancias[idx]:.4f}")

# PASO 5: Visualización de resultados
print("\n📊 PASO 5: VISUALIZACIÓN DE RESULTADOS")
print("-" * 30)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Matriz de confusión - Árbol
sns.heatmap(matriz_confusion_arbol, annot=True, fmt='d', cmap='Blues', 
            xticklabels=clases, yticklabels=clases, ax=axes[0,0])
axes[0,0].set_title('Matriz de Confusión - Árbol')
axes[0,0].set_xlabel('Predicción')
axes[0,0].set_ylabel('Real')

# Plot 2: Matriz de confusión - Random Forest
matriz_confusion_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(matriz_confusion_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=clases, yticklabels=clases, ax=axes[0,1])
axes[0,1].set_title('Matriz de Confusión - Random Forest')
axes[0,1].set_xlabel('Predicción')
axes[0,1].set_ylabel('Real')

# Plot 3: Distribución de clases
distribucion_test = pd.Series(y_test).value_counts().sort_index()
axes[0,2].bar(range(len(clases)), distribucion_test.values)
axes[0,2].set_title('Distribución de Clases (Test Set)')
axes[0,2].set_xlabel('Tipo de Evento')
axes[0,2].set_ylabel('Frecuencia')
axes[0,2].set_xticks(range(len(clases)))
axes[0,2].set_xticklabels(clases, rotation=45)

# Plot 4: Importancia de features
axes[1,0].barh(range(len(features)), modelo_rf.feature_importances_[indices_ord[::-1]])
axes[1,0].set_yticks(range(len(features)))
axes[1,0].set_yticklabels([features[i] for i in indices_ord[::-1]])
axes[1,0].set_title('Importancia de Features (RF)')
axes[1,0].set_xlabel('Importancia')

# Plot 5: Comparación de precisión por clase
precision_arbol, recall_arbol, _, _ = precision_recall_fscore_support(y_test, y_pred_arbol, average=None)
precision_rf, recall_rf, _, _ = precision_recall_fscore_support(y_test, y_pred_rf, average=None)

x_pos = np.arange(len(clases))
width = 0.35

axes[1,1].bar(x_pos - width/2, precision_arbol, width, label='Árbol', alpha=0.7)
axes[1,1].bar(x_pos + width/2, precision_rf, width, label='Random Forest', alpha=0.7)
axes[1,1].set_xlabel('Tipo de Evento')
axes[1,1].set_ylabel('Precisión')
axes[1,1].set_title('Precisión por Clase')
axes[1,1].set_xticks(x_pos)
axes[1,1].set_xticklabels(clases, rotation=45)
axes[1,1].legend()

# Plot 6: Árbol de decisión simplificado
plot_tree(modelo_arbol, max_depth=3, feature_names=features, 
          class_names=clases, filled=True, ax=axes[1,2])
axes[1,2].set_title('Árbol de Decisión (profundidad 3)')

plt.tight_layout()
plt.savefig('resultados_clasificacion.png', dpi=100, bbox_inches='tight')
print("📈 Gráficos guardados como 'resultados_clasificacion.png'")

# PASO 6: Análisis de costos
print("\n💰 PASO 6: ANÁLISIS DE COSTOS DE CLASIFICACIÓN ERRÓNEA")
print("-" * 30)

# Definir costos por tipo de error (en USD)
costos_error = {
    'falla_bomba': {
        'no_detectado': 500000,  # Costo de no detectar una falla real
        'falsa_alarma': 10000    # Costo de falsa alarma
    },
    'obstruccion': {
        'no_detectado': 200000,
        'falsa_alarma': 5000
    },
    'fuga_menor': {
        'no_detectado': 100000,
        'falsa_alarma': 3000
    },
    'mantenimiento_preventivo': {
        'no_detectado': 50000,
        'falsa_alarma': 8000
    },
    'ajuste_parametros': {
        'no_detectado': 20000,
        'falsa_alarma': 2000
    },
    'alarma_falsa': {
        'no_detectado': 5000,
        'falsa_alarma': 1000
    }
}

def calcular_costo_total(y_true, y_pred, clases, costos):
    """Calcula el costo total de clasificación errónea"""
    matriz = confusion_matrix(y_true, y_pred)
    costo_total = 0
    
    for i, clase_real in enumerate(clases):
        for j, clase_pred in enumerate(clases):
            if i != j:  # Error de clasificación
                if i in [0, 2, 4]:  # Eventos críticos no detectados
                    if j == 1:  # Clasificado como mantenimiento
                        costo = costos[clase_real]['no_detectado'] * 0.5
                    else:
                        costo = costos[clase_real]['no_detectado']
                else:  # Falsa alarma
                    costo = costos[clase_real]['falsa_alarma']
                
                costo_total += matriz[i, j] * costo
    
    return costo_total

# Calcular costos para ambos modelos
costo_arbol = calcular_costo_total(y_test, y_pred_arbol, clases, costos_error)
costo_rf = calcular_costo_total(y_test, y_pred_rf, clases, costos_error)

print(f"💸 Costo total de errores:")
print(f"   Árbol de Decisión: ${costo_arbol:,.0f}")
print(f"   Random Forest: ${costo_rf:,.0f}")
print(f"   Ahorro con Random Forest: ${costo_arbol - costo_rf:,.0f}")

# PASO 7: Sistema de alertas inteligente
print("\n🚨 PASO 7: SISTEMA DE ALERTAS INTELIGENTE")
print("-" * 30)

def generar_alerta_inteligente(probabilidades, clases, umbrales_custom=None):
    """
    Genera alertas basadas en probabilidades y umbrales adaptativos
    """
    umbrales_default = {
        'falla_bomba': 0.3,
        'obstruccion': 0.4,
        'fuga_menor': 0.5,
        'mantenimiento_preventivo': 0.6,
        'ajuste_parametros': 0.7,
        'alarma_falsa': 0.8
    }
    
    umbrales = umbrales_custom if umbrales_custom else umbrales_default
    
    alertas = []
    for i, prob_vector in enumerate(probabilidades):
        max_prob_idx = np.argmax(prob_vector)
        max_prob = prob_vector[max_prob_idx]
        clase_predicha = clases[max_prob_idx]
        
        umbral = umbrales.get(clase_predicha, 0.5)
        
        if max_prob >= umbral:
            nivel_criticidad = 'CRÍTICA' if clase_predicha in ['falla_bomba', 'obstruccion'] else 'MEDIA'
            alertas.append({
                'evento_id': i,
                'tipo_evento': clase_predicha,
                'probabilidad': max_prob,
                'nivel': nivel_criticidad,
                'accion_recomendada': f'Verificar {clase_predicha} inmediatamente'
            })
    
    return alertas

# Generar alertas para los datos de prueba
alertas = generar_alerta_inteligente(y_pred_proba_rf, clases)

print(f"🚨 Alertas generadas: {len(alertas)}")
print(f"\nPrimeras 5 alertas:")
for alerta in alertas[:5]:
    print(f"   [{alerta['nivel']}] {alerta['tipo_evento']} "
          f"(prob: {alerta['probabilidad']:.3f})")

# PASO 8: Evaluación temporal
print("\n📅 PASO 8: EVALUACIÓN DE PATRONES TEMPORALES")
print("-" * 30)

# Analizar distribución de eventos por hora y día de la semana
df_test = df_clasificacion.loc[X_test.index].copy()
df_test['prediccion'] = le.inverse_transform(y_pred_rf)
df_test['es_correcto'] = (df_test['tipo_evento'] == df_test['prediccion'])

print(f"📊 Precisión por hora del día:")
precision_por_hora = df_test.groupby('hora')['es_correcto'].mean()
hora_mejor = precision_por_hora.idxmax()
hora_peor = precision_por_hora.idxmin()
print(f"   Mejor hora: {hora_mejor}:00 ({precision_por_hora[hora_mejor]:.3f})")
print(f"   Peor hora: {hora_peor}:00 ({precision_por_hora[hora_peor]:.3f})")

print(f"\n📊 Precisión por día de la semana:")
dias_semana = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
precision_por_dia = df_test.groupby('dia_semana')['es_correcto'].mean()
for dia_num, precision in precision_por_dia.items():
    print(f"   {dias_semana[dia_num]}: {precision:.3f}")

# PASO 9: Recomendaciones operacionales
print("\n🎯 PASO 9: RECOMENDACIONES OPERACIONALES")
print("-" * 30)

# Análizar qué features son más críticas para cada tipo de evento
print(f"🔍 Análisis de criticidad por tipo de evento:")

for i, clase in enumerate(clases):
    # Encontrar casos donde se predijo esta clase correctamente
    mask = (y_test == i) & (y_pred_rf == i)
    if mask.sum() > 0:
        # Características promedio para esta clase
        X_clase = X_test[mask]
        valores_promedio = X_clase.mean()
        
        # Features más distintivas (mayor desviación de la media global)
        desviaciones = abs(valores_promedio - X_train.mean()) / X_train.std()
        top_features = desviaciones.sort_values(ascending=False).head(3)
        
        print(f"\n   {clase}:")
        for feature, desv in top_features.items():
            valor_promedio = valores_promedio[feature]
            print(f"      {feature}: {valor_promedio:.2f} (desv: {desv:.2f})")

# RESUMEN FINAL
print("\n" + "="*55)
print("📋 RESUMEN DEL ANÁLISIS DE CLASIFICACIÓN")
print("="*55)

print(f"🏆 Mejor modelo: Random Forest")
print(f"   Precisión general: {accuracy_rf:.3f}")
print(f"   Costo de errores: ${costo_rf:,.0f}")

print(f"\n💡 Insights operacionales:")
feature_mas_importante = features[indices_ord[0]]
print(f"   • Variable más crítica: {feature_mas_importante}")
print(f"   • Eventos más difíciles de detectar: {clases[precision_rf.argmin()]}")
print(f"   • Mayor precisión en horario: {hora_mejor}:00")

print(f"\n🚨 Sistema de alertas:")
print(f"   • Alertas generadas: {len(alertas)}")
print(f"   • Eventos críticos detectados: {sum(1 for a in alertas if a['nivel'] == 'CRÍTICA')}")

print(f"\n🎯 Recomendaciones de implementación:")
print(f"   1. Implementar umbrales adaptativos por tipo de evento")
print(f"   2. Monitorear más de cerca durante {dias_semana[precision_por_dia.idxmin()]}")
print(f"   3. Validar alertas de {clases[precision_rf.argmin()]} manualmente")
print(f"   4. Reentrenar modelo mensualmente")

print(f"\n✅ Demo completado. Todos los conceptos de ML cubiertos!")
print(f"💡 Siguiente paso: Implementar en laboratorios prácticos")