""" This is a simple content based suggestion script
    Computes pairwise cosine similarity scores for all movies based on their plot 
    Descriptions and recommend movies based on that similairty score threshold"""

# Import Pandas
# Assign it as the alias pd
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)