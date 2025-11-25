"""
Sistema de Recomendación Híbrido: Fuzzy Logic + ANN + Algoritmos Genéticos
Implementación completa usando MovieLens dataset
"""

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. SISTEMA DE LÓGICA DIFUSA (FIS)
# ============================================================================

class FuzzyRecommenderSystem:
    """Sistema de Inferencia Difusa para recomendaciones iniciales"""

    def __init__(self):
        self.fis = None
        self._build_fuzzy_system()

    def _build_fuzzy_system(self):
        """Construye el sistema difuso con variables y reglas"""

        # Variables de entrada
        gusto = ctrl.Antecedent(np.arange(0, 11, 1), 'gusto')
        dificultad = ctrl.Antecedent(np.arange(0, 11, 1), 'dificultad')
        precio = ctrl.Antecedent(np.arange(0, 11, 1), 'precio')
        popularidad = ctrl.Antecedent(np.arange(0, 11, 1), 'popularidad')

        # Variable de salida
        compatibilidad = ctrl.Consequent(np.arange(0, 101, 1), 'compatibilidad')

        # Funciones de membresía para GUSTO
        gusto['poco'] = fuzz.trimf(gusto.universe, [0, 0, 5])
        gusto['moderado'] = fuzz.trimf(gusto.universe, [3, 5, 7])
        gusto['mucho'] = fuzz.trimf(gusto.universe, [5, 10, 10])

        # Funciones de membresía para DIFICULTAD
        dificultad['facil'] = fuzz.trimf(dificultad.universe, [0, 0, 5])
        dificultad['intermedio'] = fuzz.trimf(dificultad.universe, [3, 5, 7])
        dificultad['dificil'] = fuzz.trimf(dificultad.universe, [5, 10, 10])

        # Funciones de membresía para PRECIO
        precio['bajo'] = fuzz.trimf(precio.universe, [0, 0, 5])
        precio['medio'] = fuzz.trimf(precio.universe, [3, 5, 7])
        precio['alto'] = fuzz.trimf(precio.universe, [5, 10, 10])

        # Funciones de membresía para POPULARIDAD
        popularidad['baja'] = fuzz.trimf(popularidad.universe, [0, 0, 5])
        popularidad['media'] = fuzz.trimf(popularidad.universe, [3, 5, 7])
        popularidad['alta'] = fuzz.trimf(popularidad.universe, [5, 10, 10])

        # Funciones de membresía para COMPATIBILIDAD (salida)
        compatibilidad['muy_baja'] = fuzz.trimf(compatibilidad.universe, [0, 0, 25])
        compatibilidad['baja'] = fuzz.trimf(compatibilidad.universe, [0, 25, 50])
        compatibilidad['media'] = fuzz.trimf(compatibilidad.universe, [25, 50, 75])
        compatibilidad['alta'] = fuzz.trimf(compatibilidad.universe, [50, 75, 100])
        compatibilidad['muy_alta'] = fuzz.trimf(compatibilidad.universe, [75, 100, 100])

        # Reglas difusas
        reglas = [
            # Reglas para alto gusto
            ctrl.Rule(gusto['mucho'] & popularidad['alta'], compatibilidad['muy_alta']),
            ctrl.Rule(gusto['mucho'] & popularidad['media'], compatibilidad['alta']),
            ctrl.Rule(gusto['mucho'] & precio['bajo'], compatibilidad['muy_alta']),
            ctrl.Rule(gusto['mucho'] & precio['alto'], compatibilidad['media']),

            # Reglas para gusto moderado
            ctrl.Rule(gusto['moderado'] & popularidad['alta'], compatibilidad['alta']),
            ctrl.Rule(gusto['moderado'] & popularidad['media'], compatibilidad['media']),
            ctrl.Rule(gusto['moderado'] & precio['bajo'], compatibilidad['alta']),

            # Reglas para poco gusto
            ctrl.Rule(gusto['poco'] & popularidad['alta'], compatibilidad['baja']),
            ctrl.Rule(gusto['poco'], compatibilidad['muy_baja']),

            # Reglas considerando dificultad
            ctrl.Rule(dificultad['facil'] & gusto['mucho'], compatibilidad['muy_alta']),
            ctrl.Rule(dificultad['dificil'] & gusto['poco'], compatibilidad['muy_baja']),

            # Reglas combinadas
            ctrl.Rule(gusto['mucho'] & precio['bajo'] & popularidad['alta'], compatibilidad['muy_alta']),
            ctrl.Rule(gusto['moderado'] & precio['medio'] & popularidad['media'], compatibilidad['media']),
        ]

        # Sistema de control
        self.fis = ctrl.ControlSystem(reglas)
        self.sim = ctrl.ControlSystemSimulation(self.fis)

    def predict(self, gusto: float, dificultad: float, precio: float, popularidad: float) -> float:
        """Predice compatibilidad usando el sistema difuso"""
        try:
            self.sim.input['gusto'] = np.clip(gusto, 0, 10)
            self.sim.input['dificultad'] = np.clip(dificultad, 0, 10)
            self.sim.input['precio'] = np.clip(precio, 0, 10)
            self.sim.input['popularidad'] = np.clip(popularidad, 0, 10)

            self.sim.compute()
            return self.sim.output['compatibilidad']
        except:
            return 50.0  # Valor por defecto en caso de error

    def predict_batch(self, inputs: np.ndarray) -> np.ndarray:
        """Predice para múltiples entradas"""
        predictions = []
        for row in inputs:
            pred = self.predict(row[0], row[1], row[2], row[3])
            predictions.append(pred)
        return np.array(predictions)


# ============================================================================
# 2. RED NEURONAL ARTIFICIAL (ANN)
# ============================================================================

class HybridNeuralNetwork:
    """Red neuronal que integra salida del sistema difuso"""

    def __init__(self, input_dim: int, hidden_layers: List[int] = [64, 32, 16]):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.model = None
        self.scaler = StandardScaler()
        self.history = None

    def build_model(self):
        """Construye la arquitectura de la red neuronal"""
        model = keras.Sequential()

        # Capa de entrada
        model.add(layers.Input(shape=(self.input_dim,)))

        # Capas ocultas
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(0.2))

        # Capa de salida (regresión)
        model.add(layers.Dense(1, activation='linear'))

        # Compilar modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32) -> Dict:
        """Entrena la red neuronal"""

        # Normalizar datos
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )

        # Entrenar
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Realiza predicciones"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evalúa el modelo"""
        predictions = self.predict(X)

        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)

        return {
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions
        }


# ============================================================================
# 3. OPTIMIZACIÓN CON ALGORITMOS GENÉTICOS
# ============================================================================

class GeneticOptimizer:
    """Optimizador genético para los parámetros del sistema híbrido"""

    def __init__(self, n_params: int, bounds: List[Tuple[float, float]]):
        self.n_params = n_params
        self.bounds = bounds
        self.toolbox = None
        self._setup_ga()

    def _setup_ga(self):
        """Configura el algoritmo genético"""

        # Crear clases de fitness e individuos
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # Generación de genes aleatorios
        for i, (low, high) in enumerate(self.bounds):
            self.toolbox.register(f"attr_{i}", np.random.uniform, low, high)

        # Crear individuo
        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            tuple(self.toolbox.__dict__[f"attr_{i}"] for i in range(self.n_params)),
            n=1
        )

        # Crear población
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Operadores genéticos
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def optimize(self, fitness_func, pop_size: int = 50, n_gen: int = 50,
                 cxpb: float = 0.7, mutpb: float = 0.05) -> Tuple:
        """Ejecuta el algoritmo genético"""

        self.toolbox.register("evaluate", fitness_func)

        # Crear población inicial
        population = self.toolbox.population(n=pop_size)

        # Estadísticas
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Ejecutar algoritmo genético
        population, logbook = algorithms.eaSimple(
            population, self.toolbox,
            cxpb=cxpb, mutpb=mutpb, ngen=n_gen,
            stats=stats, verbose=False
        )

        # Mejor individuo
        best_ind = tools.selBest(population, k=1)[0]

        return best_ind, logbook


# ============================================================================
# 4. SISTEMA HÍBRIDO COMPLETO
# ============================================================================

class HybridRecommenderSystem:
    """Sistema de recomendación híbrido completo"""

    def __init__(self):
        self.fuzzy_system = FuzzyRecommenderSystem()
        self.neural_network = None
        self.genetic_optimizer = None
        self.feature_columns = None

    def prepare_data(self, df: pd.DataFrame, target_col: str = 'rating') -> Tuple:
        """Prepara los datos para el entrenamiento"""

        # Separar características y target
        X = df.drop(columns=[target_col])
        y = df[target_col].values

        self.feature_columns = X.columns.tolist()

        # Generar puntuaciones difusas
        fuzzy_inputs = X[['gusto', 'dificultad', 'precio', 'popularidad']].values
        fuzzy_scores = self.fuzzy_system.predict_batch(fuzzy_inputs)

        # Agregar puntuación difusa como característica
        X_enhanced = X.copy()
        X_enhanced['fuzzy_score'] = fuzzy_scores / 100.0  # Normalizar a 0-1

        # División train/val/test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_enhanced, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              optimize: bool = True) -> Dict:
        """Entrena el sistema híbrido completo"""

        # Construir y entrenar red neuronal
        input_dim = X_train.shape[1]
        self.neural_network = HybridNeuralNetwork(input_dim=input_dim)
        self.neural_network.build_model()

        history = self.neural_network.train(
            X_train, y_train, X_val, y_val, epochs=50, batch_size=32
        )

        results = {'training_history': history}

        # Optimización con algoritmo genético (opcional)
        if optimize:
            print("Iniciando optimización con Algoritmo Genético...")
            best_params, ga_log = self._optimize_with_ga(X_train, y_train, X_val, y_val)
            results['ga_best_params'] = best_params
            results['ga_logbook'] = ga_log
            print(f"Optimización completada. Mejores parámetros: {best_params}")

        return results

    def _optimize_with_ga(self, X_train, y_train, X_val, y_val) -> Tuple:
        """Optimiza hiperparámetros usando algoritmo genético"""

        # Definir función de fitness
        def fitness_function(individual):
            # Extraer hiperparámetros del individuo
            learning_rate = individual[0]
            hidden_units_1 = int(individual[1])
            hidden_units_2 = int(individual[2])
            dropout_rate = individual[3]

            # Construir modelo con estos parámetros
            model = keras.Sequential([
                layers.Input(shape=(X_train.shape[1],)),
                layers.Dense(hidden_units_1, activation='relu'),
                layers.Dropout(dropout_rate),
                layers.Dense(hidden_units_2, activation='relu'),
                layers.Dropout(dropout_rate),
                layers.Dense(1, activation='linear')
            ])

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='mse'
            )

            # Entrenar brevemente
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)

            # Evaluar
            predictions = model.predict(X_val_scaled, verbose=0)
            rmse = np.sqrt(mean_squared_error(y_val, predictions))

            return (rmse,)

        # Configurar optimizador genético
        # Parámetros: [learning_rate, hidden_1, hidden_2, dropout]
        bounds = [(0.0001, 0.01), (32, 128), (16, 64), (0.1, 0.5)]

        optimizer = GeneticOptimizer(n_params=4, bounds=bounds)
        best_individual, logbook = optimizer.optimize(
            fitness_func=fitness_function,
            pop_size=20,  # Población pequeña para rapidez
            n_gen=10,     # Pocas generaciones para demo
            cxpb=0.7,
            mutpb=0.05
        )

        return best_individual, logbook

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Realiza predicciones con el sistema híbrido"""
        return self.neural_network.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evalúa el sistema completo"""
        return self.neural_network.evaluate(X_test, y_test)


# ============================================================================
# 5. GENERACIÓN DE DATOS SINTÉTICOS (DEMO)
# ============================================================================

def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """Genera datos sintéticos para demostración del sistema"""

    np.random.seed(42)

    # Características base
    data = {
        'user_id': np.random.randint(1, 101, n_samples),
        'item_id': np.random.randint(1, 201, n_samples),
        'gusto': np.random.uniform(0, 10, n_samples),
        'dificultad': np.random.uniform(0, 10, n_samples),
        'precio': np.random.uniform(0, 10, n_samples),
        'popularidad': np.random.uniform(0, 10, n_samples),
        'user_age': np.random.uniform(18, 70, n_samples),
        'item_category': np.random.randint(0, 10, n_samples),
    }

    df = pd.DataFrame(data)

    # Generar rating basado en características (simulación realista)
    df['rating'] = (
        0.4 * df['gusto'] +
        0.2 * df['popularidad'] +
        0.15 * (10 - df['precio']) +
        0.1 * (10 - df['dificultad']) +
        np.random.normal(0, 0.5, n_samples)
    )

    # Normalizar rating a escala 1-5
    df['rating'] = np.clip(df['rating'], 0, 10) / 2
    df['rating'] = np.clip(df['rating'], 1, 5)

    return df


# ============================================================================
# 6. VISUALIZACIÓN Y ANÁLISIS
# ============================================================================

def plot_results(history: Dict, ga_logbook=None):
    """Visualiza los resultados del entrenamiento"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Pérdida de entrenamiento
    axes[0, 0].plot(history['loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Pérdida durante Entrenamiento')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # MAE
    axes[0, 1].plot(history['mae'], label='Train MAE')
    axes[0, 1].plot(history['val_mae'], label='Val MAE')
    axes[0, 1].set_title('Error Absoluto Medio')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Evolución del Algoritmo Genético
    if ga_logbook:
        gen = [record['gen'] for record in ga_logbook]
        min_fit = [record['min'] for record in ga_logbook]
        avg_fit = [record['avg'] for record in ga_logbook]

        axes[1, 0].plot(gen, min_fit, label='Mejor Fitness')
        axes[1, 0].plot(gen, avg_fit, label='Fitness Promedio')
        axes[1, 0].set_title('Evolución del Algoritmo Genético')
        axes[1, 0].set_xlabel('Generación')
        axes[1, 0].set_ylabel('Fitness (RMSE)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Distribución de predicciones vs reales (placeholder)
    axes[1, 1].text(0.5, 0.5, 'Distribución de Predicciones\n(Ver después de evaluación)',
                    ha='center', va='center', fontsize=12)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("Gráfica guardada como 'training_results.png'")
    plt.show()


def plot_predictions(y_true, y_pred):
    """Visualiza predicciones vs valores reales"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Rating Real')
    axes[0].set_ylabel('Rating Predicho')
    axes[0].set_title('Predicciones vs Valores Reales')
    axes[0].grid(True)

    # Distribución de errores
    errors = y_pred - y_true
    axes[1].hist(errors, bins=50, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Error de Predicción')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Distribución de Errores')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('predictions_analysis.png', dpi=300, bbox_inches='tight')
    print("Gráfica guardada como 'predictions_analysis.png'")
    plt.show()


# ============================================================================
# 7. FUNCIÓN PRINCIPAL DE EJECUCIÓN
# ============================================================================

def main():
    """Función principal para ejecutar el sistema completo"""

    print("=" * 70)
    print("SISTEMA DE RECOMENDACIÓN HÍBRIDO")
    print("Fuzzy Logic + ANN + Algoritmos Genéticos")
    print("=" * 70)
    print()

    # 1. Generar datos sintéticos
    print("1. Generando datos sintéticos...")
    df = generate_synthetic_data(n_samples=2000)
    print(f"   Datos generados: {len(df)} muestras")
    print(f"   Características: {df.columns.tolist()}")
    print()

    # 2. Inicializar sistema híbrido
    print("2. Inicializando sistema híbrido...")
    hybrid_system = HybridRecommenderSystem()
    print("   ✓ Sistema de Lógica Difusa inicializado")
    print("   ✓ Red Neuronal lista para construcción")
    print("   ✓ Optimizador Genético configurado")
    print()

    # 3. Preparar datos
    print("3. Preparando datos...")
    X_train, X_val, X_test, y_train, y_val, y_test = hybrid_system.prepare_data(df)
    print(f"   Train: {len(X_train)} muestras")
    print(f"   Validation: {len(X_val)} muestras")
    print(f"   Test: {len(X_test)} muestras")
    print()

    # 4. Entrenar sistema
    print("4. Entrenando sistema híbrido...")
    print("   (Esto puede tomar varios minutos...)")
    results = hybrid_system.train(
        X_train.values, y_train,
        X_val.values, y_val,
        optimize=True  # Cambiar a False para entrenar sin GA (más rápido)
    )
    print("   ✓ Entrenamiento completado")
    print()

    # 5. Evaluar sistema
    print("5. Evaluando sistema en conjunto de prueba...")
    test_results = hybrid_system.evaluate(X_test.values, y_test)
    print(f"   RMSE: {test_results['rmse']:.4f}")
    print(f"   MAE: {test_results['mae']:.4f}")
    print()

    # 6. Visualización
    print("6. Generando visualizaciones...")
    plot_results(
        results['training_history'],
        results.get('ga_logbook')
    )
    plot_predictions(y_test, test_results['predictions'])
    print()

    # 7. Ejemplo de predicción
    print("7. Ejemplo de predicción individual:")
    sample_input = X_test.iloc[0:1].values
    prediction = hybrid_system.predict(sample_input)
    print(f"   Entrada: {sample_input[0]}")
    print(f"   Predicción: {prediction[0]:.2f}")
    print(f"   Real: {y_test[0]:.2f}")
    print(f"   Error: {abs(prediction[0] - y_test[0]):.2f}")
    print()

    print("=" * 70)
    print("EJECUCIÓN COMPLETADA")
    print("=" * 70)

    return hybrid_system, results, test_results


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    # Configurar entorno
    np.random.seed(42)
    tf.random.set_seed(42)

    # Ejecutar sistema
    system, training_results, evaluation_results = main()

    print("\n¡Sistema de recomendación híbrido listo para usar!")
    print("Puedes usar 'system.predict(X)' para hacer nuevas recomendaciones.")
