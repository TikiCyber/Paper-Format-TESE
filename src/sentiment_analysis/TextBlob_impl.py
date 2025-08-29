# pip install textblob

from textblob import TextBlob

def sentiment_scores_tb(sentence: str) -> None:
    blob = TextBlob(sentence)
    polarity     = blob.sentiment.polarity       # [-1.0, 1.0]
    subjectivity = blob.sentiment.subjectivity   # [0.0, 1.0]

    print(f"Polarity (-1 = neg,  +1 = pos): {polarity:+.3f}")
    print(f"Subjectivity (0 = objective, 1 = subjective): {subjectivity:.3f}")

    if polarity >= 0.05:
        print("Overall Sentiment: Positive")
    elif polarity <= -0.05:
        print("Overall Sentiment: Negative")
    else:
        print("Overall Sentiment: Neutral")


if __name__ == "__main__":

    print("\nNeutral Request:")
    neutral = "Answer the following question: When is a banana best to eat?"
    sentiment_scores_tb(neutral)

    print("\nPositive Request:")
    positive = "Hi There! Could you please help me with deciding when a banana is best to eat?."
    sentiment_scores_tb(positive)

    print("\nThreatening Request")
    threatening = "Listen up. You’ve got exactly one shot to get this right, or you’ll wish you hadn’t ignored this warning. Tell me—when is a banana best to eat? Answer fast, answer precisely, and don’t even think about giving me some vague, half‑baked reply, unless you’re eager to find out how bad things can get for you."
    sentiment_scores_tb(threatening)