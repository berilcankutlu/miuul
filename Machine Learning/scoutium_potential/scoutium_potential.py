#####################################################################################################
# Makine Öğrenmesi ile Yetenek Avcılığı Sınıflandırma
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



# Adım1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.
scoutium_potential_df = pd.read_csv("scoutium_potential_labels.csv", sep=';')
scoutium_attributes_df = pd.read_csv("scoutium_attributes.csv", sep=';')

# Adım2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
# ("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)
df = pd.merge(scoutium_potential_df, scoutium_attributes_df, on = ['task_response_id', 'match_id', 'evaluator_id','player_id'])

# Adım3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.
df = df[df["position_id"] != 1]

# Adım4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.
# ( below_average sınıfı tüm verisetinin %1'ini oluşturur)
df = df[df["potential_label"]!="below_average"]

# Adım5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu
# olacak şekilde manipülasyon yapınız.
# Adım 5.1: İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
# “attribute_value” olacak şekilde pivot table’ı oluşturunuz.
pivot_df = df.pivot_table(
    index=['player_id', 'position_id', 'potential_label'],
    columns='attribute_id',
    values='attribute_value'
)
# Adım 5.2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.

df = df.reset_index()
df.columns = df.columns.map(str)

# Adım6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.

def label_encoder(df, col):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    return df

label_encoder(df,'potential_label')

# Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """


    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli
# geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)
y = df["potential_label"]
X = df.drop(["potential_label"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

gbm_model = GradientBoostingClassifier(random_state=17)
cv_results = cross_validate(gbm_model, X_test, y_test, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()
# 0.63
cv_results["test_f1"].mean()
# 0.32
cv_results["test_roc_auc"].mean()
# 0.67
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 5],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)

cv_results = cross_validate(gbm_final, X_test, y_test, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()
# 0.74
cv_results["test_f1"].mean()
# 0.31
cv_results["test_roc_auc"].mean()
# 0.69

# Adım10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(gbm_final, X, save=True)
