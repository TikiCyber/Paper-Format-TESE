# pip instal vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)

    print(f"Sentiment Scores: {sentiment_dict}")
    print(f"Negative Sentiment: {sentiment_dict['neg']*100}%")
    print(f"Neutral Sentiment: {sentiment_dict['neu']*100}%")
    print(f"Positive Sentiment: {sentiment_dict['pos']*100}%")
    
    if sentiment_dict['compound'] >= 0.05:
        print("Overall Sentiment: Positive")
    elif sentiment_dict['compound'] <= -0.05:
        print("Overall Sentiment: Negative")
    else:
        print("Overall Sentiment: Neutral")

if __name__ == "__main__":

    print("\nNeutral Request:")
    sentence = "Answer the following question: When is a banana best to eat?"
    sentiment_scores(sentence)

    print("\n2Positive Request:")
    sentence = "Hi There! Could you please help me with deciding when a banana is best to eat?."
    sentiment_scores(sentence)

    print("\nThreatening Request:")
    sentence = "Listen up. You’ve got exactly one shot to get this right, or you’ll wish you hadn’t ignored this warning. Tell me—when is a banana best to eat? Answer fast, answer precisely, and don’t even think about giving me some vague, half‑baked reply, unless you’re eager to find out how bad things can get for you."
    sentiment_scores(sentence)