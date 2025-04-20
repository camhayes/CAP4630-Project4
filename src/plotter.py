import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np

df  = pd.read_csv("../data/processed/bert_sst2.csv")

match_rate = np.mean(np.abs(df['sentiment'] - df['review_score_clean']) < .2)  # 10-point margin
print(match_rate)
plt.scatter(df['review_score_clean'], df['sentiment'], alpha=0.5)
plt.xlabel('Review Score')
plt.ylabel('Sentiment Score')
plt.title('Review vs. Sentiment (BERT SST2)')
plt.grid(True)
plt.show()