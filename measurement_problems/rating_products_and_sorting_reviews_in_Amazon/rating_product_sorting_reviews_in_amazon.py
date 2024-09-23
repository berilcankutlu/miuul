# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_csv("rating_örneği/amazon_review.csv")
df_.head()
# veri üzerinde hata yapılma ihtimaline karşı kopya oluşturuyorum
df= df_.copy()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Görev 1: Average Rating’i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Adım 1: Ürünün ortalama puanını hesaplayınız.
df["overall"].mean()

# 4.587589013224822

# Adım 2: Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.

df.loc[df["day_diff"] <= df["day_diff"].quantile(0.25), "overall"].mean()
# 4.6957928802588995
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.25)) & (df["day_diff"] <= df["day_diff"].quantile(0.50)), "overall"].mean()
# 4.636140637775961
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.50)) & (df["day_diff"] <= df["day_diff"].quantile(0.75)), "overall"].mean()
# 4.571661237785016
df.loc[df["day_diff"] > df["day_diff"].quantile(0.75), "overall"].mean()
# 4.4462540716612375

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= df["day_diff"].quantile(0.25), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(df["day_diff"] > df["day_diff"].quantile(0.25)) & (df["day_diff"] <= df["day_diff"].quantile(0.50)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(df["day_diff"] > df["day_diff"].quantile(0.50)) & (df["day_diff"] <= df["day_diff"].quantile(0.75)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[df["day_diff"] > df["day_diff"].quantile(0.75), "overall"].mean() * w4 / 100

time_based_weighted_average(df)
# 4.595593165128118

# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.
df["overall"].mean()
# 4.587589013224822

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#Görev 2: Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Adım 1: helpful_no değişkenini üretiniz.
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# Adım 2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz.

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

def score_up_down_diff(up, down):
    return up - down


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("score_pos_neg_diff", ascending=False).head(20)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("score_average_rating", ascending=False).head(20)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("wilson_lower_bound", ascending=False).head(20)

# # Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
df.sort_values("wilson_lower_bound", ascending=False).head(20)
