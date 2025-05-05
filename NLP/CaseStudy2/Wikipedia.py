#######################################################################################################################
# Wikipedia metinleri içeren veri setine metin ön işleme ve görselleştirme yapılması
#######################################################################################################################
from warnings import filterwarnings
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv("wiki_data.csv")
df = df[:2000]

def clean_text(text):
    text = text.str.lower()
    text = text.str.replace(r"[^\w\s]", "")
    text = text.str.replace("\n", "")
    text = text.str.replace("\d", "")
    return text

df["text"] = clean_text(df["text"])

def remove_stopwords(text):
    import nltk
    #nltk.download("stopwords")
    sw = stopwords.words("english")
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    return text

df["text"] = remove_stopwords(df["text"])

# frekansı 1000den az olanların silinmesi
drop = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
df["text"] = df["text"].apply(lambda x: " ".join(x for x in x.split() if x not in drop))

# tokenize
#nltk.download("punkt")
df["text"].apply(lambda x: TextBlob(x).words).head()

#lemmatization
#nltk.download("wordnet")
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending=False)

tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
plt.show()

# WordCloud
text = " ".join(i for i in df.text)
wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud.png")








#############################################################################################################
# Tüm Aşamaların Fonksiyonlaştırılması
#############################################################################################################

def text_preprocessing(text,Barplot=False, Wordcloud=False):
    text = text.str.lower()
    text = text.str.replace('[^\w\s]', '')
    text = text.str.replace("\n", '')
    text = text.str.replace('\d', '')
    sw = stopwords.words('English')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    sil = pd.Series(' '.join(text).split()).value_counts()[-1000:]
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in sil))

    if Barplot:
        tf = text.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        tf.columns = ["words", "tf"]
        tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
        plt.show()

    if Wordcloud:
        text = " ".join(i for i in text)
        wordcloud = WordCloud(max_font_size=50,
                              max_words=100,
                              background_color="white").generate(text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    return text


text_preprocess(df["text"])

text_preprocess(df["text"], True, True)





