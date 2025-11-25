"""
Métricas de evaluación avanzadas para sistemas de recomendación
Incluye: Precisión, Recall, Cobertura, Diversidad, Novelty
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class RecommenderMetrics:
    """Métricas completas para sistemas de recomendación"""

    def __init__(self, threshold: float = 3.5):
        """
        Args:
            threshold: Umbral para considerar una recomendación como relevante
        """
        self.threshold = threshold

    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))

    def precision_at_k(self, y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
        """
        Precisión@K: Proporción de ítems relevantes en top-K recomendaciones
        """
        # Obtener índices de top-K predicciones
        top_k_indices = np.argsort(y_pred)[-k:]

        # Contar cuántos son relevantes (rating >= threshold)
        relevant = np.sum(y_true[top_k_indices] >= self.threshold)

        return relevant / k

    def recall_at_k(self, y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
        """
        Recall@K: Proporción de ítems relevantes encontrados en top-K
        """
        # Total de ítems relevantes
        total_relevant = np.sum(y_true >= self.threshold)

        if total_relevant == 0:
            return 0.0

        # Índices de top-K predicciones
        top_k_indices = np.argsort(y_pred)[-k:]

        # Cuántos relevantes están en top-K
        relevant_in_top_k = np.sum(y_true[top_k_indices] >= self.threshold)

        return relevant_in_top_k / total_relevant

    def f1_score_at_k(self, y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
        """F1-Score@K"""
        precision = self.precision_at_k(y_true, y_pred, k)
        recall = self.recall_at_k(y_true, y_pred, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def ndcg_at_k(self, y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain@K
        Considera el orden de las recomendaciones
        """
        # Obtener índices ordenados por predicción
        top_k_indices = np.argsort(y_pred)[-k:][::-1]

        # DCG: suma de (2^rel - 1) / log2(i + 2)
        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            rel = y_true[idx]
            dcg += (2 ** rel - 1) / np.log2(i + 2)

        # IDCG: DCG ideal (ordenando por y_true)
        ideal_indices = np.argsort(y_true)[-k:][::-1]
        idcg = 0.0
        for i, idx in enumerate(ideal_indices):
            rel = y_true[idx]
            idcg += (2 ** rel - 1) / np.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def catalog_coverage(self, recommendations: List[List[int]],
                        total_items: int) -> float:
        """
        Cobertura del catálogo: Proporción de ítems que aparecen en recomendaciones

        Args:
            recommendations: Lista de listas con IDs de ítems recomendados
            total_items: Número total de ítems en el catálogo
        """
        unique_items = set()
        for rec_list in recommendations:
            unique_items.update(rec_list)

        return len(unique_items) / total_items

    def diversity(self, recommendations: List[List[int]],
                 item_features: np.ndarray = None) -> float:
        """
        Diversidad intra-lista: Qué tan diferentes son los ítems recomendados

        Args:
            recommendations: Lista de listas con IDs de ítems recomendados
            item_features: Matriz de características de ítems (opcional)
        """
        if item_features is None:
            # Si no hay features, usar diversidad basada en unicidad
            avg_unique_ratio = []
            for rec_list in recommendations:
                unique_ratio = len(set(rec_list)) / len(rec_list)
                avg_unique_ratio.append(unique_ratio)
            return np.mean(avg_unique_ratio)

        # Diversidad basada en distancia entre características
        diversities = []
        for rec_list in recommendations:
            if len(rec_list) < 2:
                continue

            features = item_features[rec_list]
            # Calcular distancia promedio entre todos los pares
            distances = []
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    dist = np.linalg.norm(features[i] - features[j])
                    distances.append(dist)

            if distances:
                diversities.append(np.mean(distances))

        return np.mean(diversities) if diversities else 0.0

    def novelty(self, recommendations: List[List[int]],
               item_popularity: Dict[int, int]) -> float:
        """
        Novelty: Capacidad de recomendar ítems no populares

        Args:
            recommendations: Lista de listas con IDs de ítems recomendados
            item_popularity: Diccionario {item_id: count}
        """
        total_popularity = sum(item_popularity.values())

        novelty_scores = []
        for rec_list in recommendations:
            rec_novelty = 0.0
            for item in rec_list:
                if item in item_popularity:
                    # Novelty = -log2(popularidad)
                    prob = item_popularity[item] / total_popularity
                    rec_novelty += -np.log2(prob + 1e-10)

            novelty_scores.append(rec_novelty / len(rec_list))

        return np.mean(novelty_scores)

    def serendipity(self, recommendations: List[List[int]],
                   user_history: List[List[int]],
                   y_true: List[np.ndarray]) -> float:
        """
        Serendipity: Recomendaciones relevantes pero inesperadas

        Args:
            recommendations: Lista de listas con IDs recomendados
            user_history: Historial de cada usuario
            y_true: Ratings verdaderos para cada recomendación
        """
        serendipity_scores = []

        for rec_list, history, ratings in zip(recommendations, user_history, y_true):
            unexpected = [item for item in rec_list if item not in history]

            if len(unexpected) > 0:
                # Relevancia de los ítems inesperados
                relevant_unexpected = sum(1 for item in unexpected
                                        if ratings[rec_list.index(item)] >= self.threshold)
                serendipity_scores.append(relevant_unexpected / len(rec_list))
            else:
                serendipity_scores.append(0.0)

        return np.mean(serendipity_scores)

    def evaluate_all(self, y_true: np.ndarray, y_pred: np.ndarray,
                    k_values: List[int] = [5, 10, 20]) -> Dict:
        """Calcula todas las métricas principales"""

        results = {
            'rmse': self.rmse(y_true, y_pred),
            'mae': self.mae(y_true, y_pred),
        }

        # Métricas @K
        for k in k_values:
            results[f'precision@{k}'] = self.precision_at_k(y_true, y_pred, k)
            results[f'recall@{k}'] = self.recall_at_k(y_true, y_pred, k)
            results[f'f1@{k}'] = self.f1_score_at_k(y_true, y_pred, k)
            results[f'ndcg@{k}'] = self.ndcg_at_k(y_true, y_pred, k)

        return results

    def compare_models(self, models_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compara múltiples modelos

        Args:
            models_results: {model_name: {metric: value}}
        """
        df = pd.DataFrame(models_results).T
        return df.round(4)

    def plot_metrics_comparison(self, models_results: Dict[str, Dict],
                               save_path: str = None):
        """Visualiza comparación de modelos"""

        df = self.compare_models(models_results)

        # Seleccionar métricas principales
        main_metrics = ['rmse', 'mae', 'precision@10', 'recall@10', 'f1@10', 'ndcg@10']
        available_metrics = [m for m in main_metrics if m in df.columns]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, metric in enumerate(available_metrics):
            if i < len(axes):
                df[metric].plot(kind='bar', ax=axes[i], color='skyblue', edgecolor='black')
                axes[i].set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Modelo')
                axes[i].set_ylabel('Valor')
                axes[i].grid(True, alpha=0.3)
                axes[i].tick_params(axis='x', rotation=45)

        # Ocultar ejes no usados
        for i in range(len(available_metrics), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfica guardada en: {save_path}")

        plt.show()


class BaselineComparator:
    """Compara el sistema híbrido con métodos baseline"""

    @staticmethod
    def random_recommender(y_true: np.ndarray) -> np.ndarray:
        """Recomendaciones aleatorias"""
        return np.random.uniform(1, 5, size=y_true.shape)

    @staticmethod
    def popularity_recommender(train_data: pd.DataFrame,
                              test_indices: List[int]) -> np.ndarray:
        """Recomienda basado en popularidad"""
        # Calcular rating promedio por ítem
        item_avg = train_data.groupby('item_id')['rating'].mean()

        # Para test, asignar rating promedio del ítem
        predictions = []
        for idx in test_indices:
            item_id = train_data.iloc[idx]['item_id']
            pred = item_avg.get(item_id, train_data['rating'].mean())
            predictions.append(pred)

        return np.array(predictions)

    @staticmethod
    def user_average_recommender(train_data: pd.DataFrame,
                                 test_indices: List[int]) -> np.ndarray:
        """Recomienda basado en promedio del usuario"""
        user_avg = train_data.groupby('user_id')['rating'].mean()

        predictions = []
        for idx in test_indices:
            user_id = train_data.iloc[idx]['user_id']
            pred = user_avg.get(user_id, train_data['rating'].mean())
            predictions.append(pred)

        return np.array(predictions)


def demo_evaluation():
    """Demostración de uso de las métricas"""

    print("=" * 70)
    print("DEMOSTRACIÓN DE MÉTRICAS DE EVALUACIÓN")
    print("=" * 70)
    print()

    # Generar datos de ejemplo
    np.random.seed(42)
    n_samples = 1000

    y_true = np.random.uniform(1, 5, n_samples)
    y_pred_good = y_true + np.random.normal(0, 0.5, n_samples)  # Modelo bueno
    y_pred_bad = np.random.uniform(1, 5, n_samples)  # Modelo aleatorio

    # Clipear valores
    y_pred_good = np.clip(y_pred_good, 1, 5)
    y_pred_bad = np.clip(y_pred_bad, 1, 5)

    # Inicializar evaluador
    metrics = RecommenderMetrics(threshold=3.5)

    # Evaluar modelos
    results_good = metrics.evaluate_all(y_true, y_pred_good)
    results_bad = metrics.evaluate_all(y_true, y_pred_bad)

    # Comparar
    comparison = metrics.compare_models({
        'Modelo Híbrido': results_good,
        'Modelo Aleatorio': results_bad
    })

    print("Comparación de Modelos:")
    print(comparison)
    print()

    # Visualizar
    metrics.plot_metrics_comparison({
        'Modelo Híbrido': results_good,
        'Modelo Aleatorio': results_bad
    }, save_path='metrics_comparison.png')

    print("Evaluación completada.")


if __name__ == "__main__":
    demo_evaluation()
