from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv("tweets_labeled.csv")
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

df.dropna(inplace=True)

# date kolonunun gmt+03.00 olarak değiştirilmesi
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.tz_convert('Etc/GMT-3')

# mevsim değişkeninin oluşturulması
df["month"] = df["date"].dt.month_name()
df["month"] = df["month"].replace({"December": "Aralık",
                                   'January': 'Ocak',
                                   'February': "Şubat",
                                   'March': 'Mart',
                                   'April': 'Nisan',
                                   'May': 'Mayıs',
                                   'June': 'Haziran',
                                   'July': 'Temmuz',
                                   'August': 'Ağustos',
                                   'September': 'Eylül',
                                   'October': 'Ekim',
                                   'November': 'Kasım'})
seasons = {'Ocak': 'Kış',
           'Şubat': 'Kış',
           'Mart': 'İlkbahar',
           'Nisan': 'İlkbahar',
           'Mayıs': 'İlkbahar',
           'Haziran': 'Yaz',
           'Temmuz': 'Yaz',
           'Ağustos': 'Yaz',
           'Eylül': 'Sonbahar',
           'Ekim': 'Sonbahar',
           'Kasım': 'Sonbahar',
           'Aralık': 'Kış'}
df['seasons'] = df['month'].map(seasons)

# gün değişkenin oluşturulması
df["day"] = [date.strftime("%A") for date in df["date"]]
df["day"] = df["day"].replace({"Monday" : "Pazartesi",
                               "Tuesday" : "Salı",
                               "Wednesday" : "Çarşamba",
                               "Thursday" : "Perşembe",
                               "Friday" : "Cuma",
                               "Saturday" : "Cumartesi",
                               "Sunday" : "Pazar"})

# 4 saatlik periyotların olduğu değişkenlerin oluşuturulması
df["hour"] = df["date"].dt.hour
df["4hour"] = (df["hour"] // 2) *2
interval = {0:"0-2",
            2:"2-4",
            4:"4-6",
            6:"6-8",
            8:"8-10",
            10:"10-12",
            12:"12-14",
            14:"14-16",
            16:"16-18",
            18:"18-20",
            20:"20-22",
            22:"22-24"
            }
df["4hour"] = df["4hour"].map(interval)
df["time_interval"] = df["4hour"].replace({"0-2": "22-02",
                                                   "22-24": "22-02",
                                                   "2-4": "02-06",
                                                   "4-6": "02-06",
                                                   "6-8": "06-10",
                                                   "8-10": "06-10",
                                                   "10-12": "10-14",
                                                   "12-14": "10-14",
                                                   "14-16": "14-18",
                                                   "16-18": "14-18",
                                                   "18-20": "18-22",
                                                   "20-22": "18-22"})

df.drop(["4hour", "hour"], axis=1, inplace=True)

cols = ["time_interval", "days", "seasons"]
df["label"].replace(1, value="pozitif", inplace=True)
df["label"].replace(-1, value="negatif", inplace=True)
df["label"].replace(0, value="nötr", inplace=True)

numerik = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
kategorik= df.select_dtypes(include=['object']).columns.tolist()

print("Numerik Değişkenler:", numerik)
print("Kategorik Değişkenler:", kategorik)

label_counts = df['label'].value_counts()
label_percentages = df['label'].value_counts(normalize=True) * 100

df["tweet"] = df["tweet"].str.lower()
df["label"] = LabelEncoder().fit_transform(df["label"])
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["tweet"])
y = df["label"]
log_model = LogisticRegression().fit(X,y)
cross_val_score(log_model, X, y, scoring="accuracy",cv=5).mean()

df_21 = pd.read_csv("tweets_21.csv")
df_21["tweet"] = df_21["tweet"].str.lower()
df_21 = df_21[df_21["tweet"].notna()]
X_21 = tfidf.transform(df_21["tweet"])
df_21["label"] = log_model.predict(X_21)

print(df_21["label"])




















