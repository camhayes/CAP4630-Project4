import pandas as pd
from warnings import simplefilter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

###
# This file is designed to preprocess the movie reviews and movie data. 
# 
###

def main():

    movie_df  = pd.read_csv("../../data/processed/movie_metadata.csv")
        
    print("Generating sentiment scores... This will take a few minutes")    
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    emotion_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    
    def get_emotion_scores(x):
        
        pipeline_input = x if isinstance(x, str) else str(x)
        output = emotion_pipeline(pipeline_input)[0]
        return {entry['label']: entry['score'] for entry in output}
    
    movie_df['sentiment'] = movie_df['review_content'].apply(get_emotion_scores)
    movie_df['sentiment'] = movie_df['sentiment'].apply(lambda x: compute_emotion_weight(x))

    # remove unused data 
    print("Dropping unused keys...")
    # A lot of this data is parsed by this point or isn't relevant to the training model.
    movie_df = movie_df.drop(columns=['review_content'], errors='ignore')
    
    # isolate from meta data if desired
    movie_df = movie_df[['review_score_clean','sentiment']]

    print("Writing data... This will take a few minutes")
    movie_df.to_csv("../../data/processed/distilled_roberta.csv", index=False)

def compute_emotion_weight(emotions):

    pos = sum(emotions.get(e, 0) for e in ['joy', 'surprise'])
    neg = sum(emotions.get(e, 0) for e in ['anger', 'disgust', 'fear', 'sadness'])
    return round(pos - neg, 1)  # ranges from -1 to +1


if __name__ == "__main__":
    main()
