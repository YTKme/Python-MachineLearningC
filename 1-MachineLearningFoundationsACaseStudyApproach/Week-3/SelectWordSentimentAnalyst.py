"""
Analyzing Product Sentiment (Selected Words)
"""

import graphlab

def selected_word_count(word, word_list):
    """Get Selected Word Count

    Args:
        word: The selected word
        word_list: The word list

    Returns:
        An integer representing the occurrence of the word in the word list
    """

    # Check if the word is in the word list
    if word in word_list:
        return word_list[word]
    else:
        return 0


def main():
    """Main Method
    """

    # Selected words
    selected_words = ["awesome", "great", "fantastic", "amazing", "love", "horrible", "bad", "terrible", "awful", "wow", "hate"]

    products = graphlab.SFrame("amazon_baby.gl/")

    # Build the word count vector for each review
    products["word_count"] = graphlab.text_analytics.count_words(products["review"])

    # Create column to count the selected words
    for word in selected_words:
        products[word] = products["word_count"].apply(lambda x : selected_word_count(word, x))
        print "{0} Count: {1}".format(word, products[word].sum())

    # Define positive and negative sentiment
    # Ignore all 3* reviews
    products = products[products["rating"] != 3]
    # Positive sentiment = 4* or 5* reviews
    products["sentiment"] = products["rating"] >=4

    # Train sentiment classifier
    train_data, test_data = products.random_split(.8, seed=0)
    # Create the model
    selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                               target="sentiment",
                                                               features=selected_words,
                                                               validation_set=test_data)

    # Evaluate the model
    selected_words_model.evaluate(test_data, metric="roc_curve")

    # Examine review
    diaper_champ_reviews = products[products["name"] == "Baby Trend Diaper Champ"]

    # Apply learning model to understand sentiment
    diaper_champ_reviews["predicted_sentiment"] = selected_words_model.predict(diaper_champ_reviews, output_type="probability")
    #diaper_champ_reviews = diaper_champ_reviews.sort("predicted_sentiment", ascending=False)
    diaper_champ_reviews = diaper_champ_reviews.sort("rating", ascending=False)
    print "Most Positive Review (Selected Words):\n{0}".format(diaper_champ_reviews[0:10])


# Main
if __name__ == "__main__":
    main()