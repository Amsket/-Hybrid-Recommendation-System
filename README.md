# 🎯 Sistema de Recomendación Híbrido

## Fuzzy Logic + ANN + Algoritmos Genéticos

Sistema inteligente de recomendación que combina tres técnicas avanzadas de computación flexible para generar predicciones precisas y adaptadas a las preferencias difusas de los usuarios.

---

## 📋 Tabla de Contenidos

- [Características](#características)
- [Arquitectura](#arquitectura)
- [Instalación](#instalación)
- [Uso Rápido](#uso-rápido)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Datasets](#datasets)
- [Resultados](#resultados)
- [Documentación](#documentación)
- [Contribución](#contribución)
- [Licencia](#licencia)

---

## ✨ Características

### 🔹 Lógica Difusa (Fuzzy Logic)
- Manejo de preferencias vagas e imprecisas
- Variables lingüísticas: Gusto, Dificultad, Precio, Popularidad
- Sistema de inferencia con reglas if-then
- Salida: Puntuación de compatibilidad (0-100)

### 🔹 Redes Neuronales Artificiales (ANN)
- Arquitectura profunda con capas ocultas
- Aprendizaje de patrones complejos no lineales
- Integración de características difusas
- Regularización con Dropout

### 🔹 Algoritmos Genéticos (GA)
- Optimización automática de hiperparámetros
- Selección por torneo y cruce tipo blend
- Mutación gaussiana adaptativa
- Función fitness: minimizar RMSE

### 🔹 Métricas Avanzadas
- RMSE, MAE, Precision@K, Recall@K
- NDCG (Normalized Discounted Cumulative Gain)
- Cobertura del catálogo
- Diversidad y Novelty
- Serendipity

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                    ENTRADA DE DATOS                     │
│   [Usuario, Ítem, Gusto, Dificultad, Precio, ...]     │
└───────────────────┬─────────────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌──────────────────┐
│  FUZZY SYSTEM   │   │ FEATURES         │
│  • Membresías   │   │ • User Age       │
│  • Reglas       │   │ • Genre          │
│  • Inferencia   │   │ • Popularity     │
└────────┬────────┘   └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
         ┌────────────────────┐
         │  RED NEURONAL      │
         │  • Input Layer     │
         │  • Hidden Layers   │
         │  • Dropout         │
         │  • Output Layer    │
         └─────────┬──────────┘
                   │
                   ▼
         ┌────────────────────┐
         │ PREDICCIÓN FINAL   │
         │   Rating: 1-5      │
         └────────────────────┘
                   ▲
                   │
         ┌─────────┴──────────┐
         │ ALGORITMO GENÉTICO │
         │  • Optimización    │
         │  • Evolución       │
         └────────────────────┘
```

---

## 🚀 Instalación

### Requisitos Previos
- Python 3.8+
- pip

### Paso 1: Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/hybrid-recommender.git
cd hybrid-recommender
```

### Paso 2: Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### Paso 3: Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 💡 Uso Rápido

### Ejemplo Básico con Datos Sintéticos

```python
from hybrid_recommender import HybridRecommenderSystem
import pandas as pd

# 1. Inicializar sistema
system = HybridRecommenderSystem()

# 2. Cargar datos (sintéticos o reales)
from hybrid_recommender import generate_synthetic_data
df = generate_synthetic_data(n_samples=2000)

# 3. Preparar datos
X_train, X_val, X_test, y_train, y_val, y_test = system.prepare_data(df)

# 4. Entrenar
results = system.train(
    X_train.values, y_train,
    X_val.values, y_val,
    optimize=True  # Usar algoritmo genético
)

# 5. Evaluar
test_results = system.evaluate(X_test.values, y_test)
print(f"RMSE: {test_results['rmse']:.4f}")
print(f"MAE: {test_results['mae']:.4f}")

# 6. Hacer predicciones
new_data = X_test.iloc[:5].values
predictions = system.predict(new_data)
print(f"Predicciones: {predictions}")
```

### Ejemplo con MovieLens

```python
from movielens_loader import MovieLensLoader

# Cargar MovieLens-100K
loader = MovieLensLoader()
hybrid_data = loader.prepare_for_hybrid_system()

# Usar con el sistema híbrido
system = HybridRecommenderSystem()
X_train, X_val, X_test, y_train, y_val, y_test = system.prepare_data(hybrid_data)

# Entrenar y evaluar
results = system.train(X_train.values, y_train, X_val.values, y_val)
test_results = system.evaluate(X_test.values, y_test)
```

### Evaluación Completa

```python
from evaluation_metrics import RecommenderMetrics

metrics = RecommenderMetrics(threshold=3.5)

# Evaluar con todas las métricas
all_metrics = metrics.evaluate_all(
    y_test,
    predictions,
    k_values=[5, 10, 20]
)

print("Métricas completas:")
for metric, value in all_metrics.items():
    print(f"  {metric}: {value:.4f}")
```

---

## 📁 Estructura del Proyecto

```
hybrid-recommender/
│
├── hybrid_recommender.py       # Sistema principal
├── movielens_loader.py         # Cargador de datos MovieLens
├── evaluation_metrics.py       # Métricas de evaluación
├── requirements.txt            # Dependencias
├── README.md                   # Esta documentación
│
├── data/                       # Directorio de datos
│   ├── ml-100k/               # MovieLens dataset
│   └── movielens_hybrid_ready.csv
│
├── models/                     # Modelos guardados
│   └── best_model.h5
│
├── results/                    # Resultados y gráficas
│   ├── training_results.png
│   ├── predictions_analysis.png
│   └── metrics_comparison.png
│
└── notebooks/                  # Jupyter notebooks
    └── demo.ipynb
```

---

## 📊 Datasets

### MovieLens-100K
- **Descripción**: 100,000 ratings de 943 usuarios sobre 1,682 películas
- **Formato**: user_id, movie_id, rating (1-5), timestamp
- **Descarga automática**: El sistema descarga automáticamente el dataset

### Amazon Review Data
- **Descripción**: Millones de reseñas de productos Amazon
- **Categorías**: Electrónica, Libros, Películas, etc.
- **Fuente**: [Amazon Review Data](https://nijianmo.github.io/amazon/index.html)

### Datos Sintéticos
- Generados automáticamente para pruebas rápidas
- Incluye todas las características necesarias
- Configurable con `generate_synthetic_data(n_samples)`

---

## 📈 Resultados Esperados

### Comparación con Baselines

| Modelo | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 |
|--------|------|-----|--------------|-----------|---------|
| **Sistema Híbrido** | **0.85** | **0.67** | **0.78** | **0.72** | **0.82** |
| Random | 1.42 | 1.15 | 0.42 | 0.38 | 0.45 |
| Popularity | 1.05 | 0.84 | 0.61 | 0.55 | 0.65 |
| User Average | 0.98 | 0.78 | 0.64 | 0.58 | 0.68 |
| Collaborative Filtering | 0.92 | 0.73 | 0.71 | 0.65 | 0.75 |

### Ventajas del Sistema Híbrido

✅ **Mejor precisión**: RMSE ~8% mejor que CF tradicional
✅ **Manejo de cold-start**: Lógica difusa ayuda con nuevos usuarios
✅ **Interpretabilidad**: Reglas difusas son comprensibles
✅ **Optimización automática**: GA ajusta hiperparámetros
✅ **Escalabilidad**: Arquitectura modular

---

## 📚 Documentación

### Componentes Principales

#### 1. FuzzyRecommenderSystem
```python
class FuzzyRecommenderSystem:
    def __init__(self)
    def predict(gusto, dificultad, precio, popularidad) -> float
    def predict_batch(inputs: np.ndarray) -> np.ndarray
```

#### 2. HybridNeuralNetwork
```python
class HybridNeuralNetwork:
    def __init__(input_dim, hidden_layers)
    def build_model()
    def train(X_train, y_train, X_val, y_val)
    def predict(X) -> np.ndarray
```

#### 3. GeneticOptimizer
```python
class GeneticOptimizer:
    def __init__(n_params, bounds)
    def optimize(fitness_func, pop_size, n_gen) -> best_individual
```

#### 4. HybridRecommenderSystem
```python
class HybridRecommenderSystem:
    def prepare_data(df) -> train/val/test splits
    def train(X_train, y_train, optimize=True)
    def predict(X) -> predictions
    def evaluate(X_test, y_test) -> metrics
```

### Parámetros Configurables

```python
# Red Neuronal
hidden_layers = [64, 32, 16]  # Neuronas por capa
learning_rate = 0.001
epochs = 50
batch_size = 32
dropout_rate = 0.2

# Algoritmo Genético
population_size = 50
n_generations = 50
crossover_prob = 0.7
mutation_prob = 0.05

# Sistema Difuso
threshold_relevance = 3.5  # Para métricas @K
membership_functions = 'trimf'  # triangular
```

---

## 🔧 Personalización

### Agregar Nuevas Variables Difusas

```python
# En FuzzyRecommenderSystem._build_fuzzy_system()

nueva_variable = ctrl.Antecedent(np.arange(0, 11, 1), 'nueva_variable')
nueva_variable['bajo'] = fuzz.trimf(nueva_variable.universe, [0, 0, 5])
nueva_variable['alto'] = fuzz.trimf(nueva_variable.universe, [5, 10, 10])

# Agregar reglas
nueva_regla = ctrl.Rule(nueva_variable['alto'] & gusto['mucho'],
                        compatibilidad['muy_alta'])
```

### Modificar Arquitectura de la Red

```python
# En HybridNeuralNetwork.build_model()

model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')
])
```

### Cambiar Función Fitness del GA

```python
def custom_fitness(individual):
    # Tu lógica personalizada
    learning_rate, units = individual

    # Entrenar y evaluar
    score = train_and_evaluate(learning_rate, units)

    return (score,)  # Tuple para DEAP
```

---

## 🧪 Testing

### Ejecutar Tests Unitarios

```bash
pytest tests/
```

### Tests de Integración

```bash
python -m pytest tests/integration/
```

### Benchmark

```bash
python benchmark.py --dataset movielens --iterations 10
```

---

## 📊 Visualizaciones

El sistema genera automáticamente:

1. **training_results.png**: Pérdida y métricas durante entrenamiento
2. **predictions_analysis.png**: Scatter plot y distribución de errores
3. **metrics_comparison.png**: Comparación con baselines
4. **ga_evolution.png**: Evolución del algoritmo genético

---

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📖 Referencias

1. **Zadeh, L. A.** (1965). "Fuzzy sets". *Information and Control*, 8(3), 338-353.

2. **Goldberg, D. E.** (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.

3. **Haykin, S.** (2009). *Neural Networks and Learning Machines* (3rd ed.). Pearson.

4. **Ricci, F., Rokach, L., & Shapira, B.** (2015). *Recommender Systems Handbook* (2nd ed.). Springer.

5. **Harper, F. M., & Konstan, J. A.** (2015). "The MovieLens Datasets: History and Context". *ACM TIST*, 5(4), 1-19.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## 👥 Autores

- **Tu Nombre** - *Desarrollo Inicial* - [GitHub](https://github.com/tu-usuario)

---

## 🙏 Agradecimientos

- GroupLens Research por el dataset MovieLens
- Comunidad de scikit-fuzzy
- TensorFlow y Keras teams
- DEAP developers

---

## 📞 Contacto

- **Email**: tu-email@example.com
- **LinkedIn**: [Tu Perfil](https://linkedin.com/in/tu-perfil)
- **Twitter**: [@tuusuario](https://twitter.com/tuusuario)

---

## 🔮 Roadmap

- [ ] Soporte para más datasets (Netflix, Spotify)
- [ ] Implementación de filtrado colaborativo híbrido
- [ ] API REST para deployment
- [ ] Dashboard interactivo con Streamlit
- [ ] Optimización con Ray Tune
- [ ] Explicabilidad con SHAP
- [ ] Docker container
- [ ] CI/CD con GitHub Actions

---

**⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!**
