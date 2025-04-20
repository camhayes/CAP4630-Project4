import nltk
import pandas as pd
from warnings import simplefilter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

nltk.download('vader_lexicon')

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

###
# This file is designed to preprocess the movie reviews and movie data. 
# 
###

def main():

    movie_df  = pd.read_csv("../../data/processed/movie_metadata.csv")
        
    print("Generating sentiment scores... This will take a few minutes")    
    sid = SentimentIntensityAnalyzer()

    movie_df['sentiment'] = movie_df['review_content'].apply(lambda x: sid.polarity_scores(str(x))['compound'])

    movie_df['review_score_clean'] = round((movie_df['review_score_clean'] / 10) * 2 - 1, 2) # normalize the review score so that it's closer to the sentiment value

    # remove unused data 
    print("Dropping unused keys...")
    # A lot of this data is parsed by this point or isn't relevant to the training model.
    movie_df = movie_df.drop(columns=['review_content'], errors='ignore')
    
    # isolate from meta data if desired
    movie_df = movie_df[['review_score_clean','sentiment']]
    
    # Remove noise from sentiment document    
    threshold = .5 # determined to be a viable threshold for removing noise but not killing data pool
    movie_df['score_diff'] = abs(movie_df['sentiment'] - movie_df['review_score_clean'])
    movie_df = movie_df[movie_df['score_diff'] <= threshold]
    movie_df = movie_df.drop(columns=['score_diff'])
    print(len(movie_df))

    print("Writing data... This will take a few minutes")
    movie_df.to_csv("../../data/processed/vader.csv", index=False)

    plt.scatter(movie_df['review_score_clean'], movie_df['sentiment'], alpha=0.5)
    plt.xlabel('Review Score')
    plt.ylabel('Sentiment Score')
    plt.title('Review vs. Sentiment')
    plt.grid(True)
    plt.show()

def compute_emotion_weight(emotions):

    pos = sum(emotions.get(e, 0) for e in ['joy'])
    neg = sum(emotions.get(e, 0) for e in ['anger', 'disgust', 'fear', 'sadness'])
    return round(pos - neg, 1)  # ranges from -1 to +1


if __name__ == "__main__":
    main()
