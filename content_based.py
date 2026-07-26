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
print("These are the overviews of the first 5 movie plots\n", x)

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as "the"
tfidf = TfidfVectorizer(stop_words= "english")

# Replace NaN with an empty string 
metadata['overview'] = metadata['overview'].fillna("")

# Construct the required TF-IDF matrix by fitting and transforming the data 
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

# Output the shape of the tfidf_matrix
x = tfidf_matrix.shape
print("This is the shape of the tfidf_matrix\n", x)

# Array mapping from feature integer indices to the name
ary = tfidf.get_feature_names_out()[5000:5010]
print("The array mapping of the indices by name \n", ary)