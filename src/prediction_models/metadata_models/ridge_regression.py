import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


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
    print("Loading file...")
    film_df = pd.read_csv("../../../data/processed/movie_metadata.csv", header=0)
    
    # remove reviews with no tomatometer_rating
    film_df.replace('', np.nan, inplace=True)
    film_df = film_df.dropna(subset=['tomatometer_rating'])

    print("Splitting data sets...")
    X = film_df.drop(columns=['tomatometer_rating','review_content'], axis=1)
    y = film_df['tomatometer_rating']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Ridge(alpha=1.0)  # alpha can be tuned
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    y_pred  = model.predict(X_test).ravel()   # flatten [[5.8]] → [5.8]
    y_true  = y_test.values if hasattr(y_test, "values") else y_test

    r2 = r2_score(y_test, y_pred)
    print("R²:", r2)

    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual Review Score')
    plt.ylabel('Predicted Review Score')
    plt.title('Ridge Regression (Isolated Metadata): Predicted vs Actual')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
