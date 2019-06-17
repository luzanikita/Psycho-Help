import numpy as np
import pandas as pd
import pickle

from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer


def get_sequences(tokenizer, x, sentence_length):
    sequences = tokenizer.texts_to_sequences(x)

    return pad_sequences(sequences, maxlen=sentence_length)


def fix_arr(x):
    return np.append([x[0]], list(x[1:]), axis=0)


def main():
    text_tweets = pd.read_csv('../data/tweets_data.csv', delimiter='\t')

    X = text_tweets.Text.values

    print('All data has been loaded')

    SENTENCE_LENGTH = 167
    NUM = 100000

    tokenizer = Tokenizer(num_words=NUM)
    tokenizer.fit_on_texts(X)

    X_seq = get_sequences(tokenizer, X, SENTENCE_LENGTH)

    print('Input data has been tokenized')

    with open('../output/model.pkl', 'rb') as file:
        model = pickle.load(file)

    model.load_weights('../output/cnn-frozen-embeddings-37.hdf5')

    print('Model has been loaded')

    classes = np.array(['anger', 'happiness', 'love', 'neutral', 'sadness'])
    predictions = model.predict(X_seq)
    predicted_ix = np.apply_along_axis(lambda x: np.argmax(x), 1, predictions)

    text_tweets['class_prediction'] = pd.Series(np.apply_along_axis(lambda x: classes[x], 0, predicted_ix))

    text_tweets.to_csv('../output/predictions.csv', sep='\t')

    print('Predictions have been saved')


if __name__ == '__main__':
    main()
