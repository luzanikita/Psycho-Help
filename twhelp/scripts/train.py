import numpy as np
import pandas as pd

import logging
import multiprocessing
import pickle
from gensim.models import Word2Vec

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import concatenate
from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import metrics
from keras.layers import Dense, Input, Embedding, Dropout
from sklearn.model_selection import StratifiedShuffleSplit


def load_glove_model(glove_file):
    print("Loading Glove Model")
    f = open(glove_file,'r', encoding="utf-8")
    model = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        model[word] = embedding

    print("Done.",len(model)," words loaded!")
    return model


def get_sequences(tokenizer, x, sentence_length):
    sequences = tokenizer.texts_to_sequences(x)

    return pad_sequences(sequences, maxlen=sentence_length)


def load_embedding_matrix(embedding_file, tokenizer):
    embed_size = 100

    embeddings_index = dict()
    f = open(embedding_file, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    nb_words = len(tokenizer.word_index)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    embedded_count = 0
    for word, i in tokenizer.word_index.items():
        i -= 1
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            embedded_count += 1

    print('total embedded:', embedded_count, 'common words')
    del embeddings_index

    return embedding_matrix


def fix_arr(x):
    return np.append([x[0]], list(x[1:]), axis=0)


def main():
    text_tweets = pd.read_csv('../input/text_tweets.csv', delimiter='\t', index_col=0)

    X = text_tweets.content.values
    y = np.load('../input/labels.npy')

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(X, text_tweets.sentiment.values):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    X_train = fix_arr(X_train)
    y_train = fix_arr(y_train)
    X_test = fix_arr(X_test)
    y_test = fix_arr(y_test)

    print('Train and test samples have been created')

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    data = text_tweets.content.values
    model = Word2Vec(
        data, size=200, window=5, min_count=3,
        workers=multiprocessing.cpu_count())
    model.save('../output/model.w2v')

    SENTENCE_LENGTH = text_tweets.content.map(len).max()
    NUM = 100000
    DIM = 200

    tokenizer = Tokenizer(num_words=NUM)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = get_sequences(tokenizer, X_train, SENTENCE_LENGTH)
    X_test_seq = get_sequences(tokenizer, X_test, SENTENCE_LENGTH)

    tweet_input = Input(shape=(SENTENCE_LENGTH,), dtype='int32')
    tweet_encoder = Embedding(
        NUM, DIM, input_length=SENTENCE_LENGTH,
        trainable=False)(tweet_input)

    branches = []
    x = Dropout(0.2)(tweet_encoder)

    for size, filters_count in [(2, 10), (3, 10), (4, 10), (5, 10)]:
        for i in range(filters_count):
            branch = Conv1D(
                filters=1, kernel_size=size,
                padding='valid', activation='relu')(x)
            branch = GlobalMaxPooling1D()(branch)
            branches.append(branch)

    x = concatenate(branches, axis=1)
    x = Dropout(0.2)(x)
    x = Dense(30, activation='relu')(x)
    output = Dense(5, activation='softmax')(x)

    model = Model(inputs=[tweet_input], outputs=[output])
    model.compile(
        loss='binary_crossentropy', optimizer='adam',
        metrics=[metrics.categorical_accuracy])

    model.load_weights('../output/cnn-frozen-embeddings-37.hdf5')

    checkpoint = ModelCheckpoint(
        "../output/cnn-frozen-embeddings-{epoch:02d}.hdf5",
        save_best_only=True, mode='max', period=1)
    
    print(y_train.shape)
    model.fit(
        X_train_seq, y_train,
        batch_size=4, epochs=100,
        validation_data=(X_test_seq, y_test),
        callbacks=[checkpoint])

    with open('../output/model.pkl', 'wb') as file:
        pickle.dump(model, file)

    print('Model has been saved')

if __name__ == '__main__':
    main()
