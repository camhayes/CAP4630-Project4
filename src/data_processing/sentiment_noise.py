import pandas as pd
from warnings import simplefilter
import matplotlib.pyplot as plt

# Remove noise from sentiment scores where the delta between the sentiment and review is large.

df = pd.read_csv("../../data/processed/distilled_roberta.csv")

# normalize this data similar to the sentiment score, -1 -> 1
df['normalized_review_score'] = (df['review_score_clean'] / 10) * 2 - 1


threshold = .3 

df['score_diff'] = abs(df['sentiment'] - df['normalized_review_score'])

for t in [0.2, 0.3, 0.4, 0.5]:
    filtered = df[df['score_diff'] <= t]
    print(f"Threshold {t}: {len(filtered)} rows")

plt.hist(df['score_diff'], bins=50)
plt.xlabel("Sentiment - Review Score Diff")
plt.ylabel("Frequency")
plt.title("Distribution of Sentiment Alignment")
plt.show()
