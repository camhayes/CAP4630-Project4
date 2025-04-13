from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import nltk

nltk.download('vader_lexicon')

###
# This file is designed to preprocess the movie reviews and movie data. 
# 
###

def main():
    reviews_df = pd.read_csv("../data/rotten_tomatoes_critic_reviews.csv")
    movies_df = pd.read_csv("../data/rotten_tomatoes_movies.csv")
    merged_df = pd.merge(reviews_df, movies_df, on='rotten_tomatoes_link', how='inner')

    merged_df['review_score_clean'] = merged_df['review_score'].apply(clean_score)
    merged_df = merged_df.dropna(subset=['review_score_clean'])
    # make sure I cite this in my final project: https://www.nltk.org/api/nltk.sentiment.vader.html
    ###
    # TLDR: Vader analyzes and scores the sentimentality of a text based on a trained model. 
    # Words are generally positive, negative or neutral. This tool calculates and sums each text review to give me a score I can use to help predict.
    ###
    sid = SentimentIntensityAnalyzer()

    print("Parsing data... this may take a few minutes")    
    merged_df['sentiment'] = merged_df['review_content'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
    # remove unused data
    merged_df = merged_df.drop(columns=['publisher_name', 'review_date', 'content_rating', 'original_release_date', 'streaming_release_date', 'audience_status', 'audience_rating', 'audience_count'], errors='ignore')
    print("Writing data to CSV... this may take a few minutes")
    merged_df.to_csv("../data/processed/critics_movies.csv", index=False)

def clean_score(raw_score):
    # Normalize scores since there's a lot of variety in how an individual critic scores a movie. 
    if pd.isnull(raw_score):
        return None

    raw_score = str(raw_score).strip()

    # Handle fractions
    if '/' in raw_score:
        try:
            num, denom = raw_score.split('/')
            return round((float(num) / float(denom)) * 10, 2)
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
