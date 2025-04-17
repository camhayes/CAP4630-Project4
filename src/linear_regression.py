import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
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
    film_df = pd.read_csv("../data/processed/critics_movies.csv", header=0)
    print("Splitting data sets...")
    X = film_df.drop(columns=['review_score_clean','sentiment'], axis=1)
    y = film_df['review_score_clean']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = models.Sequential([
        layers.Dense(1, input_shape=(X_train.shape[1],))  # Linear output layer
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {mae:.2f}")

    y_pred = model.predict(X_test)
    
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

    y_pred  = model.predict(X_test).ravel()   # flatten [[5.8]] → [5.8]
    y_true  = y_test.values if hasattr(y_test, "values") else y_test

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"MAE  : {mae:0.3f}")
    print(f"RMSE : {rmse:0.3f}")
    print(f"R²   : {r2:0.3f}")

    from sklearn.dummy import DummyRegressor
    dum = DummyRegressor(strategy="mean").fit(X_train, y_train)
    mae_base = mean_absolute_error(y_test, dum.predict(X_test))
    print(f"Baseline MAE: {mae_base:0.3f}")

    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual Review Score')
    plt.ylabel('Predicted Review Score')
    plt.title('Predicted vs Actual')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
