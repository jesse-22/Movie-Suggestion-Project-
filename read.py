""" This is a simple Python Suggestion Sytem """

# Import Pandas
# Assign it as the alias pd
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Store the first three rows
md = metadata.head(3)

# Print the first three rows of the dataset
# Display the information to the terminal
print("First three datasets\n", md)

# Calculate the mean rating of a movie on IMDB
# Based on scale 1-10
c = metadata['vote_average'].mean()
print("Average rating for a movie is:\n", c)

# Calculate the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.90)
print("Minimum number of votes required to be considered:\n",m)

# Filter out all qualified movies into a new DataFrame
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
new_dataframe = q_movies.shape
print("New dataframe:\n", new_dataframe)

# Function that computes the weighted rating of each movie 
def weighted_rating(x, m=m, c=c):
     v = x['vote_count']
     r = x['vote_average']
     # Calculate based on the IMDB formula 
     return (v/(v+m) * r) + (m/(m+v) *c)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(20))


