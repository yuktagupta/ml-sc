import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.preprocessing import StandardScaler 
# Step 1: Load dataset 
df = pd.read_csv('C:\\Users\\Admin\\OneDrive\\Desktop\\pract\\ratings.csv')
# Step 2: Create user-item interaction matrix 
interaction_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0) 
# Step 3: Normalize the data (optional but helps with similarity calculation) 
scaler = StandardScaler(with_mean=False) 
interaction_matrix_scaled = scaler.fit_transform(interaction_matrix) 
# Step 4: Compute user-user similarity 
user_similarity = cosine_similarity(interaction_matrix_scaled) 
user_similarity_df = pd.DataFrame(user_similarity, index=interaction_matrix.index, 
columns=interaction_matrix.index) 
# Step 5: Generate recommendations 
def recommend(user_id, k=5): 
    # Find similar users 
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:k+1] 
    # Collect weighted ratings from similar users 
    similar_users_ratings = interaction_matrix.loc[similar_users.index] 
    weighted_ratings = similar_users_ratings.T.dot(similar_users) 
    # Exclude movies already rated by the user 
    user_rated = interaction_matrix.loc[user_id] 
    recommendations = weighted_ratings[user_rated == 0].sort_values(ascending=False).head(k) 
    return recommendations.index.tolist() 
# Example: Recommend movies for user ID  
user_id = int(input("Enter your input")) 
recommendations = recommend(user_id) 
print(f"Recommendations for User {user_id}: {recommendations}") 