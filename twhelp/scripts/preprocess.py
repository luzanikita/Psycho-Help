import numpy as np
import pandas as pd
import re
import os

from sklearn.preprocessing import OneHotEncoder


def load_data(data_path):
    text_tweets = pd.read_csv(
        os.path.join(data_path, 'text_emotion.csv'))

    return text_tweets


def merge_emotions(text_tweets):
    text_tweets.drop(['tweet_id', 'author'], axis=1, inplace=True)

    text_tweets.loc[text_tweets.sentiment == 'worry', 'sentiment'] = 'sadness'
    text_tweets.loc[text_tweets.sentiment == 'boredom', 'sentiment'] = 'sadness'
    text_tweets.loc[text_tweets.sentiment == 'fun', 'sentiment'] = 'happiness'
    text_tweets.loc[text_tweets.sentiment == 'relief', 'sentiment'] = 'happiness'
    text_tweets.loc[text_tweets.sentiment == 'enthusiasm', 'sentiment'] = 'happiness'
    text_tweets.loc[text_tweets.sentiment == 'surprise', 'sentiment'] = 'happiness'
    text_tweets.loc[text_tweets.sentiment == 'hate', 'sentiment'] = 'anger'

    text_tweets.drop(text_tweets[text_tweets.sentiment == 'empty'].index, axis=0, inplace=True)

    return text_tweets


def norm(arr):
    min_ = min(arr)
    max_ = max(arr)

    return 2 * (arr - min_) / (max_ - min_) - 1


def tweet_to_vector(
    tweet, words, regex,
    vocab_dict, vocab_size
):
    
    vector = np.zeros(vocab_size, dtype=np.float_)
    for w in re.findall(regex, tweet):
        if w in vocab_dict:
            vector[vocab_dict[w]] += words.iloc[vocab_dict[w]].happy_norm_avg

    return vector


def save(save_path, text_tweets):
    text_tweets.to_csv(os.path.join(save_path, 'text_tweets.csv'), sep='\t')

    np.save('../input/labels', fix_arr(text_tweets.labels.values))


def fix_arr(x):
    return np.append([x[0]], list(x[1:]), axis=0)


def main():
    text_tweets = load_data('../data')

    print('All data has been loaded')

    text_tweets = merge_emotions(text_tweets)

    print('Emotions have been merged')

    onehot_encoder = OneHotEncoder(sparse=False, dtype=int)
    integer_encoded = text_tweets.sentiment.values.reshape(len(text_tweets.sentiment), 1)
    onehot_encoder.fit(integer_encoded)
    text_tweets['labels'] = text_tweets.sentiment.apply(lambda x: onehot_encoder.transform([[x]])[0])

    print('Labels have been encoded')

    save('../input', text_tweets)

    print('All data has been saved')


if __name__ == '__main__':
    main()