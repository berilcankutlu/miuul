import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from sklearn.model_selection import cross_val_score, train_test_split,  GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from warnings import filterwarnings

filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: "%2f" %x)

df = pd.read_excel("amazon.xlsx")
###################################################################################################################
# 1. Metin Ön İşleme
##################################################################################################################
# Metindeki harflerin küçültülmesi
df["Review"] = df["Review"].str.lower()

# Sayısal ifadeler ile noktalama işaretlerinin çıkarılması
df["Review"] = df["Review"].str.replace("[^\w\s]", '')
df["Review"] = df["Review"].str.replace("\d", '')

# Bilgi içermeyen kelimelerin çıkarılması
import nltk
nltk.download('stopwords')
sw = stopwords.words("english")
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# 1000'den az geçen kelimelerin çıkarılması
drops = pd.Series(" ".join(df["Review"]).split()).value_counts()[-1000:]
df["Review"] = df["Review"].apply(lambda x: " ".join(x for x in x.split() if x not in drops))


# Lemmatization : Kelimeleri köklerine ayırma
nltk.download("wordnet")
df["Review"] = df["Review"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


###################################################################################################################
#2. Metin Görselleştirme
###################################################################################################################
# Terim frekanslarının hesaplanması
tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending=False)

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

# WordCloud
text = " ".join(i for i in df.Review)
wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud.png")

######################################################################################################################
# 3. Duygu Analizi
######################################################################################################################
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])
df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
df["Sentiment_Label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

#####################################################################################################################
# 4. ML Hazırlıkı
#####################################################################################################################
train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                    df["Sentiment_Label"],
                                                    random_state=42)

tf_idf_vectorizer = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_vectorizer.transform(test_x)

#####################################################################################################################
# 5. Lojistik regresyon
######################################################################################################################
log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)
y_pred = log_model.predict(x_test_tf_idf_word)
print(classification_report(y_pred, test_y))

cross_val_score(log_model,
                x_train_tf_idf_word,
                train_y,
                scoring="accuracy",
                cv=5).mean()
# %89

random_review = pd.Series(df["Review"].sample(1).values)
yeni_yorum = CountVectorizer().fit(train_x).transform(random_review)
pred = log_model.predict(yeni_yorum)
print(f'Review:  {random_review[0]} \n Prediction: {pred}')

#####################################################################################################################
# 6. RandomForest
######################################################################################################################
rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
cross_val_score(rf_model,
                x_train_tf_idf_word,
                train_y,
                scoring="accuracy",
                cv=5).mean()
# %91
