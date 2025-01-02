import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .movie-card {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('IMDBcleaned.csv')
    return df

def create_feature_matrix(df):
    # Combine relevant features for similarity calculation
    df['combined_features'] = df['Genre'] + ' ' + df['Plot'] + ' ' + df['Director'] + ' ' + df['Cast']
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    feature_matrix = tfidf.fit_transform(df['combined_features'])
    return feature_matrix

def kmeans_recommendations(feature_matrix, movie_idx, n_recommendations=10):
    n_clusters = min(20, feature_matrix.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(feature_matrix)
    
    movie_cluster = cluster_labels[movie_idx]
    cluster_movies = np.where(cluster_labels == movie_cluster)[0]
    
    similarities = feature_matrix[movie_idx].dot(feature_matrix[cluster_movies].T).toarray().flatten()
    similar_indices = cluster_movies[np.argsort(similarities)[-n_recommendations-1:-1]]
    
    return similar_indices[::-1]

def knn_recommendations(feature_matrix, movie_idx, n_recommendations=10):
    knn = NearestNeighbors(n_neighbors=n_recommendations+1, metric='cosine')
    knn.fit(feature_matrix)
    
    distances, indices = knn.kneighbors(feature_matrix[movie_idx].reshape(1, -1))
    return indices[0][1:]

def random_forest_recommendations(df, feature_matrix, movie_idx, n_recommendations=10):
    le = LabelEncoder()
    df['genre_encoded'] = le.fit_transform(df['Genre'].apply(lambda x: x.split(',')[0]))
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(feature_matrix.toarray(), df['genre_encoded'])
    
    predictions = rf.predict_proba(feature_matrix.toarray())
    target_genre_probs = predictions[movie_idx]
    
    similarities = np.dot(predictions, target_genre_probs)
    similar_indices = np.argsort(similarities)[-n_recommendations-1:-1]
    
    return similar_indices[::-1]

def display_movie_card(movie):
    st.markdown(f"""
        <div class="movie-card">
            <h3>{movie['Movie Name']} ({movie['Release Year']})</h3>
            <p><strong>Director:</strong> {movie['Director']}</p>
            <p><strong>Genre:</strong> {movie['Genre']}</p>
            <p><strong>IMDB Rating:</strong> {movie['IMDB Rating']} ({movie['Number of votes']} votes)</p>
            <p><strong>Cast:</strong> {movie['Cast']}</p>
            <p><strong>Plot:</strong> {movie['Plot']}</p>
        </div>
    """, unsafe_allow_html=True)

def main():
    st.title("🎬 Movie Recommendation System")
    
    # Load data
    df = load_data()
    feature_matrix = create_feature_matrix(df)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Movie selection
    selected_movie = st.sidebar.selectbox(
        "Select a movie:",
        df['Movie Name'].tolist()
    )
    
    # Algorithm selection
    algorithm = st.sidebar.radio(
        "Select recommendation algorithm:",
        ('K-means Clustering', 'K-Nearest Neighbors', 'Random Forest')
    )
    
    # Number of recommendations
    n_recommendations = st.sidebar.slider(
        "Number of recommendations:",
        min_value=5,
        max_value=20,
        value=10
    )
    
    if st.sidebar.button('Get Recommendations'):
        movie_idx = df[df['Movie Name'] == selected_movie].index[0]
        
        # Display selected movie
        st.subheader("Selected Movie")
        display_movie_card(df.iloc[movie_idx])
        
        # Get recommendations
        if algorithm == 'K-means Clustering':
            similar_indices = kmeans_recommendations(feature_matrix, movie_idx, n_recommendations)
        elif algorithm == 'K-Nearest Neighbors':
            similar_indices = knn_recommendations(feature_matrix, movie_idx, n_recommendations)
        else:  # Random Forest
            similar_indices = random_forest_recommendations(df, feature_matrix, movie_idx, n_recommendations)
        
        # Display recommendations
        st.subheader("Recommended Movies")
        cols = st.columns(2)
        for idx, movie_idx in enumerate(similar_indices):
            with cols[idx % 2]:
                display_movie_card(df.iloc[movie_idx])

if __name__ == "__main__":
    main()