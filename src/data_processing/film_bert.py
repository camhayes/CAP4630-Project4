import pandas as pd
from warnings import simplefilter
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

###
# This file is designed to preprocess the movie reviews and movie data. 
# 
###

def main():

    movie_df  = pd.read_csv("../../data/processed/movie_metadata.csv")
        
    print("Generating sentiment scores... This will take a few minutes")    
    

    model_path = "C:/Users/cmhay/Documents/bert_regression_final"

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

    model.eval()

    # Predict score for a new review
    def predict_score(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits
        return logits.item()  # single float (predicted review score)

    movie_df['sentiment'] = movie_df['review_content'].apply(predict_score)

    # normalize the review score so that it's closer to the sentiment value
    movie_df['review_score_clean'] = round((movie_df['review_score_clean'] / 10) * 2 - 1, 2) 

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
    movie_df.to_csv("../../data/processed/film_bert.csv", index=False)

    plt.scatter(movie_df['review_score_clean'], movie_df['sentiment'], alpha=0.5)
    plt.xlabel('Review Score')
    plt.ylabel('Sentiment Score')
    plt.title('Review vs. Sentiment')
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()
