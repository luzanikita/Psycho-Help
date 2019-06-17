import pandas as pd
import matplotlib.pyplot as plt

classes = ['happiness', 'neutral', 'sadness', 'anger', 'love']
tweets_df = pd.read_csv('../output/predictions.csv', sep='\t', index_col=0)

tweets_df["Date"] = pd.to_datetime(tweets_df["Date"])

tweets_df_for_plot_mes = tweets_df.groupby([tweets_df["Date"].dt.year, tweets_df["Date"].dt.month, 'class_prediction']).count()
tweets_df_for_plot_mes.drop(columns="Likes", inplace=True)
tweets_df_for_plot_mes.drop(columns="Retweets", inplace=True)

happiness_rate = {}
sadness_rate = {}
df = tweets_df.groupby([tweets_df["Date"].dt.year, tweets_df["Date"].dt.month]).count()
df.drop(columns="Likes", inplace=True)
df.drop(columns="Retweets", inplace=True)
df.drop(columns="Date", inplace=True)
df.drop(columns="class_prediction", inplace=True)

for i, row in tweets_df_for_plot_mes.iterrows():
    year, month, sentiment = i
    df.loc[(year, month), sentiment] = int(row['Text'])
    if (year, month) not in happiness_rate:
        happiness_rate[(year, month)] = 0
    if (year, month) not in sadness_rate:
        sadness_rate[(year, month)] = 0
    if sentiment in ['happiness', 'love']:
        happiness_rate[(year, month)] += int(row['Text'])
    elif sentiment in ['angry', 'sadness']:
        sadness_rate[(year, month)] += int(row['Text'])

for k in happiness_rate.keys():
    if happiness_rate[k] + sadness_rate[k] == 0:
        happiness_rate[k] = 0
    else:
        happiness_rate[k] /= happiness_rate[k] + sadness_rate[k]

happiness_rate = pd.Series(list(happiness_rate.values()))
df.fillna(0, inplace=True)
print(happiness_rate)

df.reset_index(level=0, inplace=True)
df['Year'] = df['Date']
df.drop(columns="Date", inplace=True)

df.reset_index(level=0, inplace=True)
df['Month'] = df['Date']
df.drop(columns="Date", inplace=True)

df['Happiness_Rate'] = happiness_rate
df.fillna(0.5, inplace=True)

df['Count'] = df['Text']
df.drop(columns='Text', inplace=True)

min_ = df.Count.min()
max_ = df.Count.max()
div_ = (max_ - min_)

df.loc[:, classes] /= div_

plot = df.loc[:, classes].plot.bar(stacked=True)
fig = plot.get_figure()
plt.plot(df.Happiness_Rate, color='red')
plt.legend(['Happiness rate', *classes])
plt.title('Activity diagram')
fig.savefig('../output/activity.png')

pie_df = pd.DataFrame(columns=['Text'])
for class_ in classes:
    pie_df = pie_df.append(
        tweets_df.loc[tweets_df['class_prediction'] == class_, ['Text']].count(),
        ignore_index=True)

pie_df.set_index(pd.Series(classes), inplace=True)
plt.figure()
pie_df.plot(x=classes, y='Text', kind='pie', legend=False, autopct='%1.1f%%') \
    .set_title('Sentiment classes distribution')

plt.ylabel('')
plt.savefig('../output/piechart.png')