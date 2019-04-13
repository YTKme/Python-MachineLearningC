"""
Retrieving Wikipedia Articles
"""

import graphlab

def main():
    """Main Method
    """
    # Load the data
    people = graphlab.SFrame('people_wiki.gl/')

    # Get Elton John
    elton_john = people[people['name'] == 'Elton John']
    # Get the word count
    elton_john['word_count'] = graphlab.text_analytics.count_words(elton_john['text'])
    # Create a table for word count
    elton_john_count_table = elton_john[['word_count']].stack('word_count', new_column_name = ['word', 'count'])
    # Print top 3 word count
    print 'Elton John Word Count Table\nHighest Word Count:\n'
    elton_john_count_table.sort('count', ascending=False).print_rows(num_rows=3)
    # Dictionary
    #print type(elton_john['word_count'][0])
    # Get TF-IDF
    elton_john['tfidf'] = graphlab.text_analytics.tf_idf(elton_john['word_count'])
    elton_john_tfidf_table = elton_john[['tfidf']].stack('tfidf', new_column_name=['word', 'tfidf'])
    print 'Elton John TF-IDF Table\nHighest Word Count:\n'
    elton_john_tfidf_table.sort('tfidf', ascending=False).print_rows(num_rows=3)

    # Get Victoria Beckham
    victoria_beckham = people[people['name'] == 'Victoria Beckham']
    # Get the word count
    victoria_beckham['word_count'] = graphlab.text_analytics.count_words(victoria_beckham['text'])

    # Compare Elton John to Victoria Beckham
    #elton_victoria_distance = graphlab.toolkits.distances.cosine(elton_john['word_count'][0], victoria_beckham['word_count'][0])
    #print 'Distance Between Elton John and Victoria Beckham: {0}'.format(elton_victoria_distance)

    # Get Paul McCartney
    paul_mccartney = people[people['name'] == 'Paul McCartney']
    # Get the word count
    paul_mccartney['word_count'] = graphlab.text_analytics.count_words(paul_mccartney['text'])

    # Compare Elton John to Paul McCartney
    elton_paul_distance = graphlab.toolkits.distances.cosine(elton_john['word_count'][0], paul_mccartney['word_count'][0])
    print 'Distance Between Elton John and Paul McCartney: {0}'.format(elton_paul_distance)

    # Compute TF-IDF for the corpus
    people['word_count'] = graphlab.text_analytics.count_words(people['text'])
    people['tfidf'] = graphlab.text_analytics.tf_idf(people['word_count'])

    # Build nearest neighbor model
    word_count_model = graphlab.nearest_neighbors.create(people, features=['word_count'], distance='cosine', label='name')
    tfidf_model = graphlab.nearest_neighbors.create(people, features=['tfidf'], distance='cosine', label='name')

    # Get Elton John
    elton_john = people[people['name'] == 'Elton John']
    # Get Victoria Beckham
    victoria_beckham = people[people['name'] == 'Victoria Beckham']
    # Get Paul McCartney
    paul_mccartney = people[people['name'] == 'Paul McCartney']

    # Compare Elton John to Victoria Beckham
    elton_victoria_distance = graphlab.toolkits.distances.cosine(elton_john['tfidf'][0], victoria_beckham['tfidf'][0])
    print 'Distance Between Elton John and Victoria Beckham: {0}'.format(elton_victoria_distance)

    # Compare Elton John to Paul McCartney
    elton_paul_distance = graphlab.toolkits.distances.cosine(elton_john['tfidf'][0], paul_mccartney['tfidf'][0])
    print 'Distance Between Elton John and Paul McCartney: {0}'.format(elton_paul_distance)

    # Query
    print 'Word Count Model:\n{0}'.format(word_count_model.query(elton_john))
    print 'TF-IDF Model:\n{0}'.format(tfidf_model.query(elton_john))
    # Query
    print 'Word Count Model:\n{0}'.format(word_count_model.query(victoria_beckham))
    print 'TF-IDF Model:\n{0}'.format(tfidf_model.query(victoria_beckham))
    

# Main
if __name__ == "__main__":
    main()