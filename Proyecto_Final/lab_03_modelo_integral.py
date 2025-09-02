"""
Laboratorio 03: Modelo Integral de ML para Operaciones Petroleras
==================================================================

Objetivo: Combinar predicción y clasificación para crear un sistema integral
          de monitoreo y predicción para operaciones petroleras.

Tareas:
1. Integrar datos de múltiples fuentes
2. Crear pipeline de ML completo
3. Implementar predicción de producción Y detección de eventos
4. Generar dashboard de resultados
5. Crear sistema de recomendaciones automatizado

Tiempo estimado: 45 minutos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, classification_report
import warnings
warnings.filterwarnings('ignore')

print("LABORATORIO 03: MODELO INTEGRAL DE ML")
print("=" * 55)

# TAREA 1: Integración de datos
# ==============================
print("\nTAREA 1: Integrar datos de múltiples fuentes")
print("-" * 40)

# TODO: Cargar y combinar datasets
# df_produccion = pd.read_csv('../datos/produccion_historica.csv')
# df_eventos = pd.read_csv('../datos/eventos_operacionales.csv')
# df_parametros = pd.read_csv('../datos/parametros_pozos.csv')

# TODO: Realizar merge de los datasets
# Usar fecha y well_id como claves
df_integrado = None  # Resultado del merge

print(f"Dataset integrado: {df_integrado.shape if df_integrado is not None else 'No creado'}")

# TODO: Crear features temporales
def crear_features_temporales(df):
    """
    Crea features basadas en tiempo
    """
    df = df.copy()
    # TODO: Extraer componentes de fecha
    # df['dia_semana'] = pd.to_datetime(df['fecha']).dt.dayofweek
    # df['mes'] = pd.to_datetime(df['fecha']).dt.month
    # df['trimestre'] = pd.to_datetime(df['fecha']).dt.quarter
    
    # TODO: Crear features de ventanas móviles
    # df['prod_promedio_7d'] = df['produccion_oil_bbl'].rolling(7).mean()
    # df['prod_std_7d'] = df['produccion_oil_bbl'].rolling(7).std()
    
    return df

# Aplicar ingeniería de features temporal
df_procesado = crear_features_temporales(df_integrado) if df_integrado else None


# TAREA 2: Pipeline de preparación
# =================================
print("\nTAREA 2: Crear pipeline de ML")
print("-" * 40)

# Definir features para cada modelo
features_produccion = [
    'presion_boca_psi', 'temperatura_f', 'dias_operacion',
    'choke_size', 'prod_promedio_7d', 'mes', 'trimestre'
]

features_eventos = [
    'temperatura', 'vibracion', 'presion', 'ruido_db',
    'dias_desde_mantenimiento', 'horas_operacion_continua'
]

# TODO: Crear pipelines para cada modelo
def crear_pipeline_produccion():
    """
    Pipeline para predicción de producción
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('modelo', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ))
    ])

def crear_pipeline_eventos():
    """
    Pipeline para clasificación de eventos
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('modelo', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        ))
    ])

# Crear los pipelines
pipeline_produccion = crear_pipeline_produccion()
pipeline_eventos = crear_pipeline_eventos()


# TAREA 3: Entrenamiento de modelos
# ==================================
print("\nTAREA 3: Entrenar modelos")
print("-" * 40)

# TODO: Preparar datos para modelo de producción
if df_procesado is not None:
    # Eliminar filas con NaN en features críticas
    df_prod = df_procesado.dropna(subset=features_produccion + ['produccion_oil_bbl'])
    X_prod = df_prod[features_produccion]
    y_prod = df_prod['produccion_oil_bbl']
    
    # Dividir datos
    X_prod_train, X_prod_test, y_prod_train, y_prod_test = train_test_split(
        X_prod, y_prod, test_size=0.2, random_state=42
    )
    
    # TODO: Entrenar modelo de producción
    # pipeline_produccion.fit(X_prod_train, y_prod_train)
    
    # TODO: Validación cruzada
    # scores = cross_val_score(pipeline_produccion, X_prod_train, y_prod_train, 
    #                          cv=5, scoring='neg_mean_absolute_error')
    # print(f"MAE promedio (CV): {-scores.mean():.2f} ± {scores.std():.2f}")

# TODO: Preparar datos para modelo de eventos
if df_procesado is not None:
    df_eventos = df_procesado.dropna(subset=features_eventos + ['tipo_evento'])
    X_eventos = df_eventos[features_eventos]
    y_eventos = df_eventos['tipo_evento']
    
    # Dividir datos
    X_eventos_train, X_eventos_test, y_eventos_train, y_eventos_test = train_test_split(
        X_eventos, y_eventos, test_size=0.2, stratify=y_eventos, random_state=42
    )
    
    # TODO: Entrenar modelo de eventos
    # pipeline_eventos.fit(X_eventos_train, y_eventos_train)


# TAREA 4: Evaluación integral
# =============================
print("\nTAREA 4: Evaluación integral del sistema")
print("-" * 40)

# TODO: Evaluar modelo de producción
print("Modelo de Producción:")
if 'X_prod_test' in locals():
    # y_pred_prod = pipeline_produccion.predict(X_prod_test)
    # mae = mean_absolute_error(y_prod_test, y_pred_prod)
    # mape = np.mean(np.abs((y_prod_test - y_pred_prod) / y_prod_test)) * 100
    # print(f"  MAE: {mae:.2f} bbl")
    # print(f"  MAPE: {mape:.2f}%")
    pass

# TODO: Evaluar modelo de eventos
print("\nModelo de Eventos:")
if 'X_eventos_test' in locals():
    # y_pred_eventos = pipeline_eventos.predict(X_eventos_test)
    # print(classification_report(y_eventos_test, y_pred_eventos))
    pass


# TAREA 5: Dashboard de resultados
# =================================
print("\nTAREA 5: Crear dashboard de resultados")
print("-" * 40)

fig = plt.figure(figsize=(16, 12))

# Layout del dashboard
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel 1: Predicción de producción (serie temporal)
ax1 = fig.add_subplot(gs[0, :2])
# TODO: Graficar predicciones vs reales para últimos 30 días
ax1.set_title('Predicción de Producción - Últimos 30 días', fontsize=12, fontweight='bold')
ax1.set_xlabel('Días')
ax1.set_ylabel('Producción (bbl)')
ax1.grid(True, alpha=0.3)

# Panel 2: Métricas de producción
ax2 = fig.add_subplot(gs[0, 2])
# TODO: Mostrar métricas clave
metricas_texto = """
MÉTRICAS DE PRODUCCIÓN
━━━━━━━━━━━━━━━━━
MAE: XXX bbl
MAPE: XX.X%
R²: X.XXX

Predicción promedio:
XXXX bbl/día

Tendencia: ↑ +X.X%
"""
ax2.text(0.1, 0.5, metricas_texto, fontsize=10, family='monospace')
ax2.set_title('KPIs de Producción', fontsize=12, fontweight='bold')
ax2.axis('off')

# Panel 3: Matriz de confusión de eventos
ax3 = fig.add_subplot(gs[1, 0])
# TODO: Mostrar matriz de confusión
ax3.set_title('Clasificación de Eventos', fontsize=12, fontweight='bold')

# Panel 4: Distribución de eventos predichos
ax4 = fig.add_subplot(gs[1, 1])
# TODO: Pie chart o barplot de tipos de eventos
ax4.set_title('Distribución de Eventos', fontsize=12, fontweight='bold')

# Panel 5: Importancia de features
ax5 = fig.add_subplot(gs[1, 2])
# TODO: Mostrar top 5 features más importantes
ax5.set_title('Features Más Importantes', fontsize=12, fontweight='bold')

# Panel 6: Análisis de tendencias
ax6 = fig.add_subplot(gs[2, :2])
# TODO: Mostrar tendencias de producción y eventos
ax6.set_title('Análisis de Tendencias', fontsize=12, fontweight='bold')
ax6.set_xlabel('Tiempo')
ax6.grid(True, alpha=0.3)

# Panel 7: Recomendaciones
ax7 = fig.add_subplot(gs[2, 2])
recomendaciones = """
RECOMENDACIONES
━━━━━━━━━━━━━━
✓ Optimizar choke size
  Impacto: +5% prod.

⚠ Mantenimiento preventivo
  Pozo #3 en 48hrs

✓ Ajustar presión
  Target: 1450 psi

⚠ Revisar sensor temp.
  Lecturas anómalas
"""
ax7.text(0.1, 0.5, recomendaciones, fontsize=9, family='monospace')
ax7.set_title('Acciones Recomendadas', fontsize=12, fontweight='bold')
ax7.axis('off')

plt.suptitle('DASHBOARD INTEGRAL - SISTEMA ML PETROLERO', fontsize=16, fontweight='bold')
plt.savefig('dashboard_ml_integral.png', dpi=100, bbox_inches='tight')
print("Dashboard guardado como 'dashboard_ml_integral.png'")


# TAREA 6: Sistema de recomendaciones
# ====================================
print("\nTAREA 6: Sistema de recomendaciones automatizado")
print("-" * 40)

def generar_recomendaciones(predicciones_prod, predicciones_eventos, parametros_actuales):
    """
    Genera recomendaciones basadas en las predicciones
    
    Parámetros:
    -----------
    predicciones_prod : array
        Predicciones de producción
    predicciones_eventos : array
        Predicciones de eventos
    parametros_actuales : dict
        Parámetros operacionales actuales
    
    Retorna:
    --------
    list : Lista de recomendaciones priorizadas
    """
    recomendaciones = []
    
    # TODO: Analizar tendencia de producción
    if len(predicciones_prod) > 7:
        tendencia = np.polyfit(range(7), predicciones_prod[-7:], 1)[0]
        if tendencia < -10:  # Caída significativa
            recomendaciones.append({
                'prioridad': 'ALTA',
                'tipo': 'PRODUCCIÓN',
                'accion': 'Revisar parámetros operacionales',
                'impacto': f'Caída de {abs(tendencia):.1f} bbl/día detectada'
            })
    
    # TODO: Analizar eventos predichos
    eventos_criticos = ['falla_bomba', 'obstruccion', 'fuga']
    for evento in predicciones_eventos:
        if evento in eventos_criticos:
            recomendaciones.append({
                'prioridad': 'CRÍTICA',
                'tipo': 'MANTENIMIENTO',
                'accion': f'Intervención inmediata - {evento} detectado',
                'impacto': 'Prevenir parada no planificada'
            })
    
    # TODO: Optimización de parámetros
    if parametros_actuales.get('presion_boca_psi', 0) < 1200:
        recomendaciones.append({
            'prioridad': 'MEDIA',
            'tipo': 'OPTIMIZACIÓN',
            'accion': 'Aumentar presión de boca',
            'impacto': 'Potencial aumento de 3-5% en producción'
        })
    
    return sorted(recomendaciones, key=lambda x: 
                 {'CRÍTICA': 0, 'ALTA': 1, 'MEDIA': 2}.get(x['prioridad'], 3))

# TODO: Probar el sistema de recomendaciones
parametros_ejemplo = {
    'presion_boca_psi': 1100,
    'temperatura_f': 185,
    'choke_size': 30
}

predicciones_prod_ejemplo = np.array([1000, 980, 960, 940, 920, 900, 880])
predicciones_eventos_ejemplo = ['normal', 'normal', 'falla_bomba']

recomendaciones = generar_recomendaciones(
    predicciones_prod_ejemplo,
    predicciones_eventos_ejemplo,
    parametros_ejemplo
)

print("\nRECOMENDACIONES GENERADAS:")
print("=" * 40)
for i, rec in enumerate(recomendaciones, 1):
    print(f"\n{i}. [{rec['prioridad']}] {rec['tipo']}")
    print(f"   Acción: {rec['accion']}")
    print(f"   Impacto: {rec['impacto']}")


# ANÁLISIS ROI
# ============
print("\n" + "=" * 55)
print("ANÁLISIS DE RETORNO DE INVERSIÓN (ROI)")
print("=" * 55)

def calcular_roi_ml(mejora_produccion_pct=5, reduccion_fallas_pct=30, 
                     costo_implementacion=50000):
    """
    Calcula el ROI de implementar el sistema ML
    """
    # Supuestos
    produccion_diaria_bbl = 1000
    precio_barril = 75
    dias_año = 365
    costo_falla_dia = 500000
    fallas_año_sin_ml = 6
    
    # Beneficios
    ganancia_produccion = (produccion_diaria_bbl * mejora_produccion_pct/100 * 
                          precio_barril * dias_año)
    
    fallas_evitadas = fallas_año_sin_ml * reduccion_fallas_pct/100
    ahorro_fallas = fallas_evitadas * costo_falla_dia
    
    beneficio_total = ganancia_produccion + ahorro_fallas
    
    # ROI
    roi = ((beneficio_total - costo_implementacion) / costo_implementacion) * 100
    payback_meses = (costo_implementacion / (beneficio_total / 12))
    
    print(f"Análisis ROI del Sistema ML:")
    print(f"  Inversión inicial: ${costo_implementacion:,}")
    print(f"  Ganancia por producción: ${ganancia_produccion:,.0f}/año")
    print(f"  Ahorro por prevención: ${ahorro_fallas:,.0f}/año")
    print(f"  Beneficio total anual: ${beneficio_total:,.0f}")
    print(f"  ROI: {roi:.1f}%")
    print(f"  Período de recuperación: {payback_meses:.1f} meses")
    
    return roi

# Calcular ROI
roi_sistema = calcular_roi_ml()


# PREGUNTA DE REFLEXIÓN
# =====================
print("\n" + "=" * 55)
print("PREGUNTAS DE REFLEXIÓN FINAL:")
print("=" * 55)
print("""
Reflexiona sobre la implementación completa:

1. ¿Cuáles son los principales desafíos para implementar este sistema en producción?
2. ¿Qué datos adicionales mejorarían significativamente los modelos?
3. ¿Cómo asegurarías que el modelo se mantenga actualizado con el tiempo?
4. ¿Qué métricas de negocio usarías para medir el éxito del sistema?
5. ¿Cómo comunicarías el valor de este sistema a la gerencia?

Escribe tus respuestas como comentarios aquí:
""")

# Tu respuesta:
# 1. Desafíos de implementación:
#    - Integración con sistemas existentes
#    - 
#    - 
#
# 2. Datos adicionales valiosos:
#    - 
#    - 
#    - 
#
# 3. Mantenimiento del modelo:
#    - 
#    - 
#
# 4. Métricas de éxito:
#    - 
#    - 
#    - 
#
# 5. Comunicación de valor:
#    - 
#    - 

print("\n" + "=" * 55)
print("¡FELICITACIONES! Has completado la Sesión 16")
print("=" * 55)
print("""
Has aprendido a:
✓ Preparar datos para Machine Learning
✓ Implementar modelos de regresión y clasificación
✓ Evaluar modelos con métricas apropiadas
✓ Crear sistemas de detección de anomalías
✓ Desarrollar pipelines integrales de ML
✓ Generar recomendaciones automatizadas
✓ Calcular el ROI de soluciones ML

Próximos pasos sugeridos:
→ Explorar modelos más avanzados (XGBoost, redes neuronales)
→ Implementar validación temporal para series de tiempo
→ Desarrollar APIs para servir los modelos
→ Crear pipelines de reentrenamiento automático
""")