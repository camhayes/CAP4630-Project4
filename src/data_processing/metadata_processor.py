from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
import pandas as pd
import numpy as np
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

###
# This file is designed to preprocess the movie reviews and movie data. 
# Uses one hot encoding for meta data values EXCEPT for critic scores
###

def main():

    reviews_df = pd.read_csv("../../data/rotten_tomatoes_critic_reviews.csv")

    # remove reviews with no review content
    reviews_df.replace('', np.nan, inplace=True)
    reviews_df = reviews_df.dropna(subset=['review_content'])
    
    # randomly drop n rows -> sentiment analysis is time consuming
    n = int(len(reviews_df) // 1.03)
    drop_indices = np.random.choice(reviews_df.index, n, replace=False)
    reviews_df = reviews_df.drop(drop_indices)

    # testing limits on how many reviews I have per movie
    reviews_df = reviews_df.groupby('rotten_tomatoes_link').head(20)


    movies_df = pd.read_csv("../../data/rotten_tomatoes_movies.csv")

    # encoding
    reviews_df['top_critic'] = reviews_df['top_critic'].replace({True: 1, False: 0})

    print("Normalizing genres...")
    movies_df['genre_list'] = movies_df['genres'].fillna("").apply(lambda x: [g.strip() for g in x.split(',')])
    mlb = MultiLabelBinarizer()
    genre_dummies = pd.DataFrame(
        mlb.fit_transform(movies_df['genre_list']),
        columns=[f"genre_{g}" for g in mlb.classes_]
    )
    movies_df = pd.concat([movies_df, genre_dummies], axis=1)
    movies_df.drop(columns=['genres', 'genre_list'], inplace=True)

    print("Parsing top directors...")
    movies_df['director_list'] = movies_df['directors'].fillna("").apply(lambda x: [a.strip() for a in x.split(',')])
    director_counter = Counter(director for sublist in movies_df['director_list'] for director in sublist)
    top_directors = [a for a, _ in director_counter.most_common(200)]
    for director in top_directors:
        movies_df[f'director_{director}'] = movies_df['director_list'].apply(lambda lst: int(director in lst))
    movies_df.drop(columns=['directors', 'director_list'], inplace=True)

    print("Parsing top writers...")
    movies_df['authors_list'] = movies_df['authors'].fillna("").apply(lambda x: [a.strip() for a in x.split(',')])
    authors_counter = Counter(author for sublist in movies_df['authors_list'] for author in sublist)
    top_authors = [a for a, _ in authors_counter.most_common(200)]
    for author in top_authors:
        movies_df[f'author_{author}'] = movies_df['authors_list'].apply(lambda lst: int(author in lst))
    movies_df.drop(columns=['authors', 'authors_list'], inplace=True)

    print("Parsing top actors...")
    movies_df['actor_list'] = movies_df['actors'].fillna("").apply(lambda x: [a.strip() for a in x.split(',')])
    actor_counter = Counter(actor for sublist in movies_df['actor_list'] for actor in sublist)
    top_actors = [a for a, _ in actor_counter.most_common(500)]
    for actor in top_actors:
        movies_df[f'actor_{actor}'] = movies_df['actor_list'].apply(lambda lst: int(actor in lst))
    movies_df.drop(columns=['actors', 'actor_list'], inplace=True)
    
    
    merged_df = pd.merge(reviews_df, movies_df, on='rotten_tomatoes_link', how='inner')
    
    print("Normalizing critic scores...")
    merged_df['review_score_clean'] = merged_df['review_score'].apply(clean_score)
    merged_df = merged_df.dropna(subset=['review_score_clean'])
    merged_df.drop(columns=['review_score'], inplace=True)

    # remove unused data 
    print("Dropping unused keys...")
    # A lot of this data is parsed by this point or isn't relevant to the training model.
    merged_df = merged_df.drop(columns=['genres',
                                        'directors',
                                        'actors',
                                        'authors',
                                        'runtime',
                                        'tomatometer_rotten_critics_count',
                                        'tomatometer_fresh_critics_count',
                                        'tomatometer_top_critics_count',
                                        'tomatometer_count',
                                        'tomatometer_status',
                                        'authors',
                                        'fv',
                                        'production_company',
                                        'publisher_name',
                                        'review_date',
                                        'content_rating',
                                        'critics_consensus',
                                        'original_release_date',
                                        'streaming_release_date',
                                        'audience_status',
                                        'audience_rating',
                                        'audience_count',
                                        'rotten_tomatoes_link',
                                        'critic_name',
                                        'review_type',
                                        'movie_title',
                                        'movie_info'], 
                                        errors='ignore')
    
    print("Writing data... This will take a few minutes")
    merged_df.to_csv("../../data/processed/movie_metadata.csv", index=False)

def clean_score(raw_score):
    # Normalize scores since there's a lot of variety in how an individual critic scores a movie. 
    if pd.isnull(raw_score):
        return None

    raw_score = str(raw_score).strip()

    # Handle fractions
    if '/' in raw_score:
        try:
            num, denom = raw_score.split('/')
            return round((float(num) / float(denom)) * 10, 1)
        except:
            return None

    # Handle letter grades
    letter_map = {
        'A+': 10.0, 'A': 9.5, 'A-': 9.0,
        'B+': 8.5, 'B': 8.0, 'B-': 7.5,
        'C+': 7.0, 'C': 6.5, 'C-': 6.0,
        'D+': 5.5, 'D': 5.0, 'D-': 4.5,
        'F': 3.0  # generous assumption
    }

    if raw_score.upper() in letter_map:
        return letter_map[raw_score.upper()]
    
    return None  # Couldn't parse, likely due to some weird score I don't want to account for

if __name__ == "__main__":
    main()
