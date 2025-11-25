"""
Cargador de datos MovieLens para el Sistema de Recomendación Híbrido
Descarga y prepara el dataset MovieLens-100K
"""

import os
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
import zipfile
from typing import Tuple, Dict

class MovieLensLoader:
    """Carga y prepara el dataset MovieLens"""
    
    def __init__(self, dataset_path: str = './data'):
        self.dataset_path = dataset_path
        self.url_100k = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
        self.ratings = None
        self.movies = None
        self.users = None
        
    def download_dataset(self):
        """Descarga el dataset MovieLens-100K"""
        
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        
        zip_path = os.path.join(self.dataset_path, 'ml-100k.zip')
        
        if not os.path.exists(zip_path):
            print("Descargando MovieLens-100K...")
            urlretrieve(self.url_100k, zip_path)
            print("Descarga completada.")
        
        # Extraer archivo
        extract_path = os.path.join(self.dataset_path, 'ml-100k')
        if not os.path.exists(extract_path):
            print("Extrayendo archivos...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_path)
            print("Extracción completada.")
        
        return extract_path
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Carga los archivos de datos"""
        
        data_path = self.download_dataset()
        
        # Cargar ratings
        ratings_path = os.path.join(data_path, 'u.data')
        self.ratings = pd.read_csv(
            ratings_path,
            sep='\t',
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        
        # Cargar información de películas
        movies_path = os.path.join(data_path, 'u.item')
        self.movies = pd.read_csv(
            movies_path,
            sep='|',
            encoding='latin-1',
            names=['movie_id', 'title', 'release_date', 'video_release_date',
                   'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                   'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                   'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                   'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        )
        
        # Cargar información de usuarios
        users_path = os.path.join(data_path, 'u.user')
        self.users = pd.read_csv(
            users_path,
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
        )
        
        print(f"Datos cargados:")
        print(f"  - {len(self.ratings)} ratings")
        print(f"  - {len(self.movies)} películas")
        print(f"  - {len(self.users)} usuarios")
        
        return self.ratings, self.movies, self.users
    
    def prepare_for_hybrid_system(self) -> pd.DataFrame:
        """Prepara los datos en formato requerido por el sistema híbrido"""
        
        if self.ratings is None:
            self.load_data()
        
        # Merge de datos
        df = self.ratings.merge(self.users, on='user_id')
        df = df.merge(self.movies[['movie_id', 'title', 'Action', 'Adventure', 
                                    'Comedy', 'Drama', 'Thriller', 'Sci-Fi']], 
                     on='movie_id')
        
        # Calcular características difusas
        
        # 1. GUSTO: Basado en géneros de la película y preferencias del usuario
        # Asumimos que el usuario tiene preferencia por ciertos géneros
        genre_columns = ['Action', 'Adventure', 'Comedy', 'Drama', 'Thriller', 'Sci-Fi']
        df['num_genres'] = df[genre_columns].sum(axis=1)
        df['gusto'] = np.random.uniform(5, 10, len(df))  # Simplificación
        
        # 2. DIFICULTAD: Basada en complejidad del género
        # Drama/Thriller = más complejo, Comedy/Action = menos complejo
        df['dificultad'] = (
            df['Drama'] * 7 + 
            df['Thriller'] * 6 + 
            df['Sci-Fi'] * 8 +
            df['Comedy'] * 3 + 
            df['Action'] * 4 +
            df['Adventure'] * 5
        )
        df['dificultad'] = df['dificultad'].clip(0, 10)
        
        # 3. PRECIO: Simulado (películas más nuevas = más caras)
        # Asumimos que el timestamp indica cuándo se vio, no fecha de lanzamiento
        df['precio'] = np.random.uniform(2, 8, len(df))
        
        # 4. POPULARIDAD: Número de ratings que tiene cada película
        movie_popularity = df.groupby('movie_id').size().reset_index(name='num_ratings')
        df = df.merge(movie_popularity, on='movie_id')
        df['popularidad'] = (df['num_ratings'] / df['num_ratings'].max() * 10).clip(0, 10)
        
        # Características adicionales
        df['user_age'] = df['age']
        
        # Codificar género como variable numérica
        df['user_gender'] = (df['gender'] == 'M').astype(int)
        
        # Número de géneros de la película
        df['movie_genres_count'] = df['num_genres']
        
        # Seleccionar columnas finales
        final_columns = [
            'user_id', 'movie_id', 'gusto', 'dificultad', 'precio', 
            'popularidad', 'user_age', 'user_gender', 'movie_genres_count', 'rating'
        ]
        
        df_final = df[final_columns].copy()
        
        # Normalizar características si es necesario
        df_final['gusto'] = df_final['gusto'].clip(0, 10)
        df_final['dificultad'] = df_final['dificultad'].clip(0, 10)
        df_final['precio'] = df_final['precio'].clip(0, 10)
        df_final['popularidad'] = df_final['popularidad'].clip(0, 10)
        
        print(f"\nDataset preparado para sistema híbrido:")
        print(f"  - Forma: {df_final.shape}")
        print(f"  - Columnas: {df_final.columns.tolist()}")
        print(f"\nEstadísticas:")
        print(df_final.describe())
        
        return df_final
    
    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del dataset"""
        
        if self.ratings is None:
            self.load_data()
        
        stats = {
            'num_users': self.ratings['user_id'].nunique(),
            'num_movies': self.ratings['movie_id'].nunique(),
            'num_ratings': len(self.ratings),
            'sparsity': 1 - (len(self.ratings) / 
                           (self.ratings['user_id'].nunique() * 
                            self.ratings['movie_id'].nunique())),
            'avg_rating': self.ratings['rating'].mean(),
            'rating_distribution': self.ratings['rating'].value_counts().to_dict()
        }
        
        return stats


def main():
    """Función de demostración"""
    
    print("=" * 70)
    print("CARGADOR DE DATOS MOVIELENS")
    print("=" * 70)
    print()
    
    # Inicializar cargador
    loader = MovieLensLoader()
    
    # Cargar datos originales
    ratings, movies, users = loader.load_data()
    print()
    
    # Mostrar estadísticas
    stats = loader.get_statistics()
    print("Estadísticas del dataset:")
    print(f"  - Usuarios: {stats['num_users']}")
    print(f"  - Películas: {stats['num_movies']}")
    print(f"  - Ratings: {stats['num_ratings']}")
    print(f"  - Sparsity: {stats['sparsity']:.2%}")
    print(f"  - Rating promedio: {stats['avg_rating']:.2f}")
    print()
    
    # Preparar para sistema híbrido
    hybrid_data = loader.prepare_for_hybrid_system()
    
    # Guardar dataset preparado
    output_path = './data/movielens_hybrid_ready.csv'
    hybrid_data.to_csv(output_path, index=False)
    print(f"\nDataset guardado en: {output_path}")
    
    return hybrid_data


if __name__ == "__main__":
    df = main()
