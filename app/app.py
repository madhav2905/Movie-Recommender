import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

svd_model = joblib.load('models/svd_model.pkl')
user_cf_model = joblib.load('models/user_cf_model.pkl')
item_cf_model = joblib.load('models/item_cf_model.pkl')
movie_features = joblib.load('models/movie_features.pkl')

movies = pd.read_csv('data/movies.csv')
popular_movies = pd.read_csv('data/popular_movies.csv')

if 'rating' not in popular_movies.columns:
    if 'avg_rating' in popular_movies.columns:
        popular_movies.rename(columns={'avg_rating': 'rating'}, inplace=True)
    else:
        ratings = pd.read_csv('data/ratings.csv')
        avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
        popular_movies = popular_movies.merge(avg_ratings, on='movieId', how='left')

st.title("\U0001F3AC Movie Rating Predictor & Recommender")

model_choice = st.selectbox("Choose a model", [
    "SVD", "User-Based CF", "Item-Based CF", "Content-Based", "Popularity-Based"
])

user_id = st.number_input("Enter User ID", min_value=1, step=1)

movie_name = st.selectbox("Select a Movie", movies['title'].values)

movie_id = movies[movies['title'] == movie_name]['movieId'].values[0]

if model_choice in ["SVD", "User-Based CF", "Item-Based CF"]:
    if st.button("Predict Rating"):
        if model_choice == "SVD":
            prediction = svd_model.predict(user_id, movie_id).est
        elif model_choice == "User-Based CF":
            prediction = user_cf_model.predict(user_id, movie_id).est
        else:
            prediction = item_cf_model.predict(user_id, movie_id).est

        st.success(f"\u2B50 Predicted Rating using {model_choice}: {prediction:.2f}")

st.markdown("---")

st.subheader("\U0001F3AF Get Movie Recommendations")
num_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)

if st.button("Recommend Similar Movies"):
    if model_choice == "Content-Based":
        similarity_matrix = cosine_similarity(movie_features)
        sim_scores = list(enumerate(similarity_matrix[movie_id]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        st.subheader(f"\U0001F3A5 Movies similar to **{movie_name}**:")
        for idx in movie_indices:
            st.write(f"**{movies.iloc[idx]['title']}**")

    elif model_choice == "Popularity-Based":
        st.subheader("\U0001F525 Most Popular Movies:")
        for _, row in popular_movies.head(num_recommendations).iterrows():
            rating = row.get('rating', row.get('avg_rating', 'N/A'))
            if isinstance(rating, (int, float)):
                st.write(f"**{row['title']}** (Average Rating: {rating:.2f})")
            else:
                st.write(f"**{row['title']}**")

    else:
        all_movie_ids = movies['movieId'].unique()
        scores = []
        for other_movie_id in all_movie_ids:
            if other_movie_id != movie_id:
                try:
                    if model_choice == "SVD":
                        pred = svd_model.predict(user_id, other_movie_id)
                    elif model_choice == "User-Based CF":
                        pred = user_cf_model.predict(user_id, other_movie_id)
                    else:
                        pred = item_cf_model.predict(user_id, other_movie_id)
                    scores.append((other_movie_id, pred.est))
                except:
                    continue

        top_n = sorted(scores, key=lambda x: x[1], reverse=True)[:num_recommendations]
        st.subheader(f"\U0001F3A5 Movies similar to **{movie_name}**:")
        for mid, rating in top_n:
            similar_title = movies[movies['movieId'] == mid]['title'].values[0]
            st.write(f"**{similar_title}** — Predicted Rating: {rating:.2f}")