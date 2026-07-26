""" This is a simple content based suggestion script
    Computes pairwise cosine similarity scores for all movies based on their plot 
    Descriptions and recommend movies based on that similairty score threshold"""

# Consolidate the imports used in the script
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load Movies Metadata
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Print the plot overviews of the first 5 movies 
x = metadata['overview'].head()
# print("These are the overviews of the first 5 movie plots\n", x)

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as "the"
tfidf = TfidfVectorizer(stop_words= "english")

# Replace NaN with an empty string 
metadata['overview'] = metadata['overview'].fillna("")

# Construct the required TF-IDF matrix by fitting and transforming the data 
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

# Output the shape of the tfidf_matrix
x = tfidf_matrix.shape
# print("This is the shape of the tfidf_matrix\n", x)

# Array mapping from feature integer indices to the name
ary = tfidf.get_feature_names_out()[5000:5010]
# print("The array mapping of the indices by name \n", ary)

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cs = cosine_sim.shape
# print("This is the shape of the cos_sim matrix\n", cs)

# Testing if formatting is correct 
y = cosine_sim[1]
# print("test\n", y) 

# Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index = metadata['title']).drop_duplicates()
""" test = indices[:10]
print("test\n", test) """

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim = cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similariy scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata["title"].iloc[movie_indices]

# Test
rec = get_recommendations("The Godfather")
print("rec test\n", rec)
