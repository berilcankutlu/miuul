# Hybrid Recommender System
# User Based Recommendation
# Görev 1: Veri Hazırlama
# Adım 1: movie, rating veri setlerini okutunuz.
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

movie_df = pd.read_csv("datasets/movie.csv")
rating_df = pd.read_csv("datasets/rating.csv")

movie_df.head()
rating_df.head()

# Adım 2: rating veri setine Id’lere ait film isimlerini ve türünü movie veri setinden ekleyiniz.
df = rating_df.merge(movie_df, how= "left", on="movieId")

# Adım3: Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini listede tutunuz ve
# veri setinden çıkartınız.

rate_counts = pd.DataFrame(df["title"].value_counts())

rare_movies = []
rare_movies = rate_counts[rate_counts["count"] < 1000].index

df = df[~df["title"].isin(rare_movies)]
df.shape
# Adım 4: index'te userID'lerin sutunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.


df_pivot = df.pivot_table(index=["userId"], columns=["title"], values="rating")
df_pivot.head()

# Adım5: Yapılan tüm işlemleri fonksiyonlaştırınız.

def create_user_movie_df():
    import pandas as pd
    movie_df = pd.read_csv("datasets/movie.csv")
    rating_df = pd.read_csv("datasets/rating.csv")
    df = rating_df.merge(movie_df, how= "left", on="movieId")
    rate_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = rate_counts[rate_counts["count"] < 1000].index
    df = df[~df["title"].isin(rare_movies)]
    df_pivot = df.pivot_table(index=["userId"], columns=["title"], values="rating")
    return df_pivot

user_movie_df = create_user_movie_df()

# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
# Adım 1: Rastgele bir kullanıcı id’si seçiniz.
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
# random user = 28941

# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.

random_user_df = user_movie_df[user_movie_df.index==random_user]
random_user_df.head()

# Adım3: Seçilen kullanıcıların oy kullandığı filmleri movies_watched adında bir listeye atayınız

movies_watched = random_user_df.columns[random_user_df.notna().any()].to_list()

# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi
# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve
# movies_watched_df adında yeni bir dataframe oluşturuyoruz.

movies_watched_df = user_movie_df[movies_watched]

# Adım 2: Her bir kullancının seçili user'in izlediği filmlerin kaçını izlediğini bilgisini
# taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head(5)

# Adım3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenlerin kullanıcı
# id’lerinden users_same_movies adında bir liste oluşturunuz.
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren
# kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.

final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df
# dataframe’i oluşturunuz.
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan)
# kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.shape
top_users.head()

# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz
top_users_ratings = top_users.merge(rating_df[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings["userId"].unique()
top_users_ratings.head()

# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

# Adım 2: Film id’sive her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturunuz.
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()

# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.
recommendation_df[recommendation_df["weighted_rating"] > 3.5]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.
movies_to_be_recommend.merge(movie_df[["movieId", "title"]])["title"][:5]

# Adım 6: Item-Based Recommendation
user = 28941
# Adım 1: movie,rating veri setlerini okutunuz.
movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/rating.csv')

# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.
movie[movie["movieId"] == movie_id]["title"].values[0]
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_df

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

# Son iki adımı uygulayan fonksiyon
def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)
# 1'den 6'ya kadar. 0'da filmin kendisi var. Onu dışarda bıraktık.
movies_from_item_based[1:6].index

# 'Intouchables (2011)',
# 'Father of the Bride (1991)',
# 'Anna and the King (1999)',
# 'Runaway Bride (1999)',
# 'Phantom of the Opera, The (2004)'
