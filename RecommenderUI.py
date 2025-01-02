# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# Enhanced CSS with modern design
st.markdown("""
    <meta charset="utf-8">        
    <style>
    .movie-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-5px);
    }
    .movie-rating {
        font-size: 1.2rem;
        color: #ff9800;
    }
    .section-header {
        background: #1E88E5;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('IMDBcleaned.csv', encoding='utf-8')
        return df
    except UnicodeDecodeError:
        # Fallback encoding if UTF-8 fails
        try:
            df = pd.read_csv('IMDBcleaned.csv', encoding='ISO-8859-1')
            return df
        except Exception as e:
            st.error(f"Error loading data with ISO-8859-1 encoding: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_feature_matrix(df):
    df['combined_features'] = df['Genre'] + ' ' + df['Plot'] + ' ' + df['Director'] + ' ' + df['Cast']
    tfidf = TfidfVectorizer(stop_words='english')
    return tfidf.fit_transform(df['combined_features'])

def get_recommendations(feature_matrix, movie_idx, df, algorithm='kmeans', n_recommendations=10, 
                       min_rating=0.0, genres=None, year_range=None):
    with st.spinner('Finding similar movies...'):
        progress_bar = st.progress(0)
        
        if algorithm == 'kmeans':
            n_clusters = min(20, feature_matrix.shape[0])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(feature_matrix)
            movie_cluster = cluster_labels[movie_idx]
            cluster_movies = np.where(cluster_labels == movie_cluster)[0]
            similarities = feature_matrix[movie_idx].dot(feature_matrix[cluster_movies].T).toarray().flatten()
            similar_indices = cluster_movies[np.argsort(similarities)[-n_recommendations*2-1:-1]]
            
        elif algorithm == 'knn':
            knn = NearestNeighbors(n_neighbors=n_recommendations*2+1, metric='cosine')
            knn.fit(feature_matrix)
            distances, indices = knn.kneighbors(feature_matrix[movie_idx].reshape(1, -1))
            similar_indices = indices[0][1:]
            
        else:  # Random Forest
            le = LabelEncoder()
            df['genre_encoded'] = le.fit_transform(df['Genre'].apply(lambda x: x.split(',')[0]))
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(feature_matrix.toarray(), df['genre_encoded'])
            predictions = rf.predict_proba(feature_matrix.toarray())
            target_genre_probs = predictions[movie_idx]
            similarities = np.dot(predictions, target_genre_probs)
            similar_indices = np.argsort(similarities)[-n_recommendations*2-1:-1]

        # Filter recommendations based on criteria
        filtered_indices = []
        for idx in similar_indices:
            movie = df.iloc[idx]
            
            # Check minimum rating
            if movie['IMDB Rating'] < min_rating:
                continue
                
            # Check genres if specified
            if genres and not any(genre.strip() in movie['Genre'] for genre in genres):
                continue
                
            # Check year range
            if year_range and not (year_range[0] <= movie['Release Year'] <= year_range[1]):
                continue
                
            filtered_indices.append(idx)
        
        progress_bar.empty()
        
        # If we don't have enough recommendations after filtering, show warning
        if len(filtered_indices) < n_recommendations:
            st.warning(f"Only {len(filtered_indices)} movies match your criteria. Consider relaxing your filters.")
            
        return filtered_indices[:n_recommendations]

def display_movie_card(movie, is_selected=False):
    # Define background color based on whether it's selected
    background_color = "linear-gradient(105deg, #82bef3, #ffffff)" if is_selected else "linear-gradient(105deg, #e57e9d, #ffffff)"
    border = "2px solid #1E88E5" if is_selected else "2px solid #e57e9d"
    
    st.markdown(f"""
        <div class="movie-card" style="background: {background_color}; border: {border};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3>{movie['Movie Name']}</h3>
                <span class="movie-rating">⭐ {movie['IMDB Rating']}</span>
            </div>
            <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;">
                <small>📅 {movie['Release Year']} | 🎭 {movie['Genre']} | 👥 {movie['Number of votes']} votes</small>
            </div>
            <p><strong>🎬 Director:</strong> {movie['Director']}</p>
            <p><strong>👨‍👩‍👦‍👦 Cast:</strong> {movie['Cast']}</p>
            <details>
                <summary>Show Plot</summary>
                <p style="padding: 1rem 0;">{movie['Plot']}</p>
            </details>
        </div>
    """, unsafe_allow_html=True)

def main():
    st.title("🎬 Advanced Movie Recommendation System")
    
    df = load_data()
    if df is None:
        return
        

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Settings")
        selected_movie = st.selectbox("Select a movie:", df['Movie Name'].tolist())
        
        algorithm = st.radio(
            "Algorithm:",
            [('K-means Clustering'), 
             ('K-Nearest Neighbors'),
             ('Random Forest')]
        )
        
        n_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
        
        with st.expander("Advanced Filters"):
            min_rating = st.slider("Minimum IMDB Rating", 0.0, 10.0, 5.0)
            genres = st.multiselect("Select Genres", df['Genre'].unique())
            year_range = st.slider("Release Year Range", 
                                 int(df['Release Year'].min()), 
                                 int(df['Release Year'].max()), 
                                 (1990, 2024))
        
        if st.button('Get Recommendations'):
            if not selected_movie:
                st.warning("Please select a movie first!")
                return
                
            movie_idx = df[df['Movie Name'] == selected_movie].index[0]
            feature_matrix = create_feature_matrix(df)
            
            with col2:
                st.subheader("Selected Movie")
                display_movie_card(df.iloc[movie_idx], is_selected=True)  # Selected movie with different background
                
                st.subheader("Recommended Movies")
                similar_indices = get_recommendations(
                    feature_matrix, 
                    movie_idx, 
                    df,
                    algorithm[0], 
                    n_recommendations,
                    min_rating=min_rating,
                    genres=genres,
                    year_range=year_range
                )
                
                for movie_idx in similar_indices:
                    display_movie_card(df.iloc[movie_idx])

if __name__ == "__main__":
    main()