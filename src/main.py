from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    # goal: try to predict review_score_clean for each row 
    # Training details:
        # Identif a few different models to train (maybe 4 models?)
        # Two studies: meta data only, meta data + sentiment
    ### meta data variables
        # genre_*
        # actor_*
        # author_*
        # director_*
        # runtime
        # top_critic
    ###
    film_df = pd.read_csv("../data/processed/critics_movies.csv")
    X = film_df.drop(columns=['critic_score', 'review_content', 'movie_title', 'movie_info', 'critics_consensus'])
    y = film_df["critic_score"] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return
    

if __name__ == "__main__":
    main()
