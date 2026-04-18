import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
data = {
    'User': ['A','A','A','B','B','C','C','D','D'],
    'Item': ['Book','Movie','Laptop','Book','Laptop','Movie','Laptop','Book','Movie'],
    'Rating': [5,4,5,4,4,5,4,3,5]
}
df = pd.DataFrame(data)
print(df)
matrix = df.pivot_table(index='User', columns='Item', values='Rating').fillna(0)
print(matrix)
similarity = cosine_similarity(matrix)
similarity_df = pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)
print(similarity_df)
def recommend(user):
    similar_users = similarity_df[user].sort_values(ascending=False)[1:]
    top_user = similar_users.index[0]
    recommended_items = matrix.loc[top_user][matrix.loc[user] == 0]
    return recommended_items.index.tolist()
print("Recommendations for User B:", recommend('B'))

