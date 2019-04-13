"""
Analyzing Product Sentiment (All Words)
"""

import graphlab

def main():
    """Main Method
    """

    # Selected words
    selected_words = ["awesome", "great", "fantastic", "amazing", "love", "horrible", "bad", "terrible", "awful", "wow", "hate"]

    # Read product review data
    products = graphlab.SFrame("amazon_baby.gl/")

    # Build the word count vector for each review
    products["word_count"] = graphlab.text_analytics.count_words(products["review"])

    # Define positive and negative sentiment
    # Ignore all 3* reviews
    products = products[products["rating"] != 3]
    # Positive sentiment = 4* or 5* reviews
    products["sentiment"] = products["rating"] >=4

    # Train sentiment classifier
    train_data, test_data = products.random_split(.8, seed=0)
    # Create the model
    sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                          target="sentiment",
                                                          features=["word_count"],
                                                          validation_set=test_data)

    # Evaluate the model
    sentiment_model.evaluate(test_data, metric="roc_curve")

    # Examine review
    diaper_champ_reviews = products[products["name"] == "Baby Trend Diaper Champ"]

    # Apply learning model to understand sentiment
    diaper_champ_reviews["predicted_sentiment"] = sentiment_model.predict(diaper_champ_reviews, output_type="probability")
    diaper_champ_reviews = diaper_champ_reviews.sort("predicted_sentiment", ascending=False)
    print "Most Positive Review (All Words):\n{0}".format(diaper_champ_reviews[0:1])

# Main
if __name__ == "__main__":
    main()