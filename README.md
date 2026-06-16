# Framework Modular de Diagnóstico Oncológico (Random Forest vs. XGBoost)

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green.svg)
![Rigor-Estadístico](https://img.shields.io/badge/Estad%C3%ADstica-Test%20de%20McNemar-purple.svg)

## 📌 Descripción del Proyecto

Este proyecto consiste en un **framework predictivo de producción** diseñado bajo el paradigma de Programación Orientada a Objetos (POO) en Python para la clasificación y diagnóstico de cáncer de seno. El sistema automatiza un pipeline robusto que procesa datos médicos, selecciona características críticas, optimiza hiperparámetros de modelos de ensamble (*Random Forest* y *XGBoost*) y realiza una auditoría estadística estricta para garantizar decisiones científicamente respaldadas.

Aunque el caso de uso es médico, la arquitectura modular desacoplada y los métodos avanzados de evaluación son **directamente transferibles a entornos financieros de alta fidelidad**, tales como el *Credit Scoring*, la evaluación de riesgo o la detección automatizada de fraudes transaccionales.

## 🚀 Características Clave y Arquitectura

El framework destaca por estar estructurado en componentes independientes y reutilizables, superando las limitaciones operativas de los entornos Jupyter Notebook convencionales:

* **Arquitectura en POO:** Código modular organizado en clases independientes de responsabilidad única (`DataPreprocessor`, `ModelTrainer`, `ModelEvaluator`, `MLReport`), facilitando su integración en arquitecturas de microservicios o pipelines de CI/CD.
* **Selección de Variables No Lineales:** Reducción de dimensionalidad inteligente mediante **Información Mutua (`mutual_info_classif`)**, aislando relaciones complejas que la correlación lineal tradicional ignora.
* **Mitigación de Desbalance de Clases:** Manejo analítico del desbalance nativo en datos clínicos a través del cálculo dinámico de pesos (*class weights*), optimizando el rendimiento sobre la clase minoritaria (positiva).
* **Optimización Exhaustiva (GridSearchCV):** Ajuste fino de hiperparámetros críticos mediante validación cruzada estratificada para mitigar el riesgo de sobreajuste (*overfitting*).
* **Diagnóstico de Sesgo-Varianza:** Implementación automatizada de **Curvas de Aprendizaje (`learning_curve`)** para monitorear visualmente la convergencia de los modelos y el comportamiento del error.
* **Validación Estadística Exigente (Test de McNemar):** Implementación de una prueba no paramétrica sobre las predicciones discordantes de los modelos. Esto permite certificar con base en un *p-valor* si la superioridad de un algoritmo es estadísticamente significativa y no fruto del azar.

## 🛠️ Requisitos e Instalación

**Clonar el repositorio:**

git clone [https://github.com/Gutierrez-Escobar-AJ/nombre-del-repo.git](https://github.com/Gutierrez-Escobar-AJ/ML-cancer-mama.git)
cd nombre-del-repo

**Crear y activar un entorno virtual (opcional pero recomendado):**

python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

**Instalar las dependencias requeridas:**

pip install -r requirements.txt

## 💻 Uso
Para ejecutar el pipeline completo, realizar el entrenamiento competitivo, visualizar las métricas en la consola y generar las curvas diagnósticas, simplemente ejecuta:

python3 src/bc_ml_pred.py

## 📊 Métricas de Evaluación Implementadas

El sistema no se limita a evaluar la precisión global (Accuracy), sino que genera un reporte exhaustivo con métricas críticas para datos desbalanceados:

* Precisión, Recall y F1-Score por clase.

* Área Bajo la Curva ROC (ROC-AUC) para medir la calidad del ordenamiento probabilístico.

* Coeficiente de Correlación de Matthews (MCC): Considerado la métrica más robusta para evaluar clasificadores binarios en datasets desbalanceados.

* Índice Kappa de Cohen: Para medir la concordancia de las predicciones corrigiendo el efecto del azar.

## 🔬 Metodología

### 1. Preprocesamiento Robusto
- **Escalado**: `RobustScaler` (resistente a outliers)
- **Selección**: Información Mutua para capturar relaciones no lineales
- **Manejo de desbalance**: Class weights y `scale_pos_weight`

### 2. Validación Cruzada Estratificada
- **5 folds** estratificados para preservar distribución de clases
- **GridSearchCV** con 3 folds internos para optimización

### 3. Optimización de Hiperparámetros

**Random Forest**
{
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample']
}

**XGBoost**

{
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

## 📚 Fuente de Datos
Breast Cancer Wisconsin (Diagnostic) Dataset
Kaggle - Yasser H
**https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset**

**Características:** 30 atributos celulares (radio, textura, perímetro, etc.)

**Objetivo:** Clasificar tumores en **Malignos (1)** o **Benignos (0)**

**Tamaño:** 569 muestras

## 🎯 Próximos Pasos y Mejoras
* Implementar Stacking (ensamble de modelos)

* Agregar calibración de probabilidades para mejora de confianza

* Implementar SHAP para explicabilidad de predicciones

* Crear API REST con FastAPI para servir el modelo

* Agregar Docker para despliegue contenerizado

* Implementar MLflow para seguimiento de experimentos

## 📊 Resultados de Ejecución

El repositorio incluye los resultados completos de una ejecución del pipeline en la carpeta `ML_MODELS/`:

ML_MODELS/
├── csv/ # Métricas y comparaciones
│ ├── best_hyperparameters.csv # Mejores hiperparámetros encontrados
│ ├── classification_reports.csv # Reportes de clasificación (CV)
│ ├── classification_reports_test.csv # Reportes de clasificación (Test)
│ ├── feature_importance_random_forest.csv # Importancia de variables - RF
│ ├── feature_importance_xgboost.csv # Importancia de variables - XGB
│ ├── mcnemar_test_cv.csv # Test de McNemar (CV)
│ ├── mcnemar_test_test.csv # Test de McNemar (Test)
│ ├── model_comparison_cv.csv # Comparación de modelos (CV)
│ └── model_comparison_test.csv # Comparación de modelos (Test)
├── logs/
│ └── ml_models_20260616_104415.log # Bitácora completa de ejecución
└── plots/
├── confusion_matrices.png # Matrices de confusión (CV)
├── confusion_matrices_test.png # Matrices de confusión (Test)
├── feature_importance_comparison.png # Comparación de importancia
├── learning_curves_random_forest.png # Curva de aprendizaje - RF
├── learning_curves_xgboost.png # Curva de aprendizaje - XGB
├── roc_curves_comparison.png # Curvas ROC (CV)
└── roc_curves_comparison_test.png # Curvas ROC (Test)

### Comparación de Rendimiento (Test Set)

| Modelo | ROC-AUC | Precisión | Recall | F1-Score | MCC | Kappa |
|--------|---------|-----------|--------|----------|-----|-------|
| **Random Forest** | 0.994 | 0.975 | 0.983 | 0.979 | 0.957 | 0.956 |
| **XGBoost** | 0.992 | 0.971 | 0.975 | 0.973 | 0.944 | 0.943 |

### Test de McNemar
- **χ² estadístico**: 0.500
- **p-valor**: 0.480
- **Conclusión**: ✅ No hay diferencia estadísticamente significativa entre los modelos (p > 0.05)

## 📝 Conclusiones y MLOps

El diseño de este framework promueve las mejores prácticas de la ingeniería de aprendizaje automático: incluye gestión profesional de bitácoras (logging), control estricto de semillas aleatorias para garantizar la reproducibilidad y control preventivo de advertencias (warnings). Es una solución robusta lista para simular flujos de trabajo de nivel empresarial.

## 👤 Autor
Gutierrez-Escobar-AJ

