# Sentiment Analysis using RNN

Dataset: <https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification>
 
### IMDB Movie reviews sentiment classification
Dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). 
Reviews have been preprocessed, and each review is encoded as a sequence of word indexes (integers). 
For convenience, words are indexed by overall frequency in the dataset, so that for instance the integer "3" encodes the 3rd most frequent word in the data. This allows for quick filtering operations such as: "only consider the top 10,000 most common words, but eliminate the top 20 most common words".

As a convention, "0" does not stand for a specific word, but instead is used to encode any unknown word.

```
num_words = n attribute loads only the top n words into the dataset.
```

```
pad_sequences = n pads sequences into same length.
It makes sure that all arrays in the list are of same size.
```

##### Information on Embedding layers can be found at <https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12>

##### LSTM Layer <http://adventuresinmachinelearning.com/keras-lstm-tutorial/>


