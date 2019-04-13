"""
Analyzing Product Sentiment
"""

import graphlab

def task_1(data, word_list):
    """Task 1
    """

    print "##########"
    print "# Task 1 #"
    print "##########"

    # Build a word_count vector for each review with all words
    data["word_count"] = graphlab.text_analytics.count_words(data["review"])
    # Create column to count the selected words
    for word in word_list:
        data[word] = data["word_count"].apply(lambda x : selected_word_count(word, x))
        print "{0} Count: {1}".format(word, data[word].sum())


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


def task_2(data, word_list):
    """Task 2
    """

    print "##########"
    print "# Task 2 #"
    print "##########"

    # Define positive and negative sentiment
    sentiment_data = data[data["rating"] != 3] # Negative
    sentiment_data["sentiment"] = sentiment_data["rating"] >=4 # Positive

    # Train the classifier
    train_data, test_data = sentiment_data.random_split(.8, seed=0)
    # Build the model with the selected words
    selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                               target="sentiment",
                                                               features=word_list,
                                                               validation_set=test_data)

    # Sort the coefficients by value
    selected_words_model["coefficients"].sort("value", ascending=False)

    print "Model Coefficients:\n{}".format(selected_words_model["coefficients"].print_rows(num_rows=12))


def task_3(data, word_list):
    """Task 3
    """

    print "##########"
    print "# Task 3 #"
    print "##########"

    # Define positive and negative sentiment
    sentiment_data = data[data["rating"] != 3] # Negative
    sentiment_data["sentiment"] = sentiment_data["rating"] >=4 # Positive

    # Train the classifier
    train_data, test_data = sentiment_data.random_split(.8, seed=0)
    # Build the model with the selected words
    selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                               target="sentiment",
                                                               features=word_list,
                                                               validation_set=test_data)

    # Sort the coefficients by value
    selected_words_model["coefficients"].sort("value", ascending=False)

    # Evaluate the selected words model against test data
    selected_words_model.evaluate(test_data, metric="roc_curve")
    # Show the model
    # Accuracy of the sentiment_model: 0.911
    # Accuracy of the selected_words_model: 0.843
    # 
    #selected_words_model.show(view="Categorical")

def task_4(data, word_list):
    """Task 4
    """

    print "##########"
    print "# Task 4 #"
    print "##########"

    product_name = "Baby Trend Diaper Champ"

    # Examine review of product
    product_review = data[data["name"] == product_name]
    #print "Product Review: {0}".format(len(product_review))

    # Define positive and negative sentiment
    sentiment_data = data[data["rating"] != 3] # Negative
    sentiment_data["sentiment"] = sentiment_data["rating"] >=4 # Positive

    # Train the classifier
    train_data, test_data = sentiment_data.random_split(.8, seed=0)
    # Build the model with the selected words
    selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                               target="sentiment",
                                                               features=word_list,
                                                               validation_set=test_data)
    # Build the model with all words
    sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                        target="sentiment",
                                                        features=["word_count"],
                                                        validation_set=test_data)

    # Apply the learned model to understand sentiment with the selected words
    product_review["predicted_sentiment"] = selected_words_model.predict(product_review, output_type="probability")
    product_review = product_review.sort("predicted_sentiment", ascending=False)
    print "Most Positive Review (Selected Words):\n{0}".format(product_review[0:1])
    # Apply the learned model to understand sentiment with all words
    product_review["predicted_sentiment"] = sentiment_model.predict(product_review, output_type="probability")
    product_review = product_review.sort("predicted_sentiment", ascending=False)
    print "Most Positive Review (All Words):\n{0}".format(product_review[0:1])


def main():
    """Main Method
    """
    products = graphlab.SFrame("amazon_baby.gl/")

    # Subset of words
    selected_words = [
        "awesome",
        "great",
        "fantastic",
        "amazing",
        "love",
        "horrible",
        "bad",
        "terrible",
        "awful",
        "wow",
        "hate"
    ]

    # Task 1
    task_1(products, selected_words)

    # Task 2
    task_2(products, selected_words)

    # Task 3
    task_3(products, selected_words)

    # Task 4
    task_4(products, selected_words)


# Main
if __name__ == "__main__":
    main()