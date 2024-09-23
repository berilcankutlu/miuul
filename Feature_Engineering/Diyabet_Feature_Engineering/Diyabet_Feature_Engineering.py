################################################
# Diabete Feature Engineering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df = pd.read_csv("datasets/diabetes.csv")
df.head()

# Görev 1 : Keşifçi Veri Analizi
# Adım 1: Genel resmi inceleyiniz.

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
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)
df.info()
# Glucose, BloodPressure, SkinThickness, Insulin ve BMI 0 olamaz.
# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(df, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
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

    """
    cat_cols = [col for col in df.columns if df[col].dtypes=='0']
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes != '0']
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and df[col].dtypes == '0']

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes!='0']
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {df.shape[0]}")
    print(f"Variables : {df.shape[1]}")
    print(f"cat_cols : {len(cat_cols)}")
    print(f"num_cols : {len(num_cols)}")
    print(f"cat_but_car : {len(cat_but_car)}")
    print(f"num_but_cat : {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
# numeric değişkenlerin analizi
def num_summary(df, num_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(df[num_col].describe(quantiles).T)
    if plot:
        df[num_col].hist(bins=20)
        plt.xlabel(num_col)
        plt.title(num_col)
        plt.show(block = True)

for col in num_cols:
    num_summary(df, col)

# kategorik değişkenlerin analizi
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(df, "Outcome")

# Adım 4: Hedef değişken analizi yapınız.
# hedef değişkene göre numerik değişkenlerin ortalaması
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

# Adım 5: Aykırı gözlem analizi yapınız.
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykırı Değer Analizi ve Baskılama İşlemi
for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in df.columns:
    print(col, check_outlier(df, col))


# Aykırı değerlere erişmek istersek:
def grab_outliers(dataframe, col, index=False):
    low, up = outlier_thresholds(dataframe, col)
    if dataframe[(dataframe[col] > up) | (dataframe[col] < low)].shape[0] > 10:
        print(dataframe[(dataframe[col] > up) | (dataframe[col] < low)].head())
    else:
        print(dataframe[(dataframe[col] > up) | (dataframe[col] < low)])
    if index:
        outlier_index = dataframe[(dataframe[col] > up) | (dataframe[col] < low)].index
        return outlier_index

grab_outliers(df, "Outcome")

# Adım 6: Eksik gözlem analizi yapınız.
def missing_value_table(dataframe, na=False):
    na_col = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_col].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_col].isnull().sum() / dataframe.shape[0]*100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss","ratio"])
    print(missing_df, end="\n")
    if na:
        return na_col
missing_value_table(df, True)

# Adım 7: Korelasyon analizi yapınız.
corr = df[num_cols].corr()

# Görev 2 : Feature Engineering
# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.

df.describe().T
# Glucose, BloodPressure, Skin Thickness, Insulin ve BMI değerleri sıfır olamayacağı için
# bu değerler eksik olup sıfırla değiştirilmiştir.
missings = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

for col in missings:
    df[col] = np.where(df[col] == 0, np.nan, df[col])
na = missing_value_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na)

for col in missings:
    df.loc[df[col].isnull(), col] = df[col].median()


df.isnull().sum()
df.describe().T


# Adım 2: Yeni değişkenler oluşturunuz.

df.head()

# yaş değişkeninden yeni değişkenler çıkaralım
df.loc[(df["Age"] < 18 ), "New_Age_Category"] = "young"
df.loc[(df["Age"] >= 18) & (df["Age"] < 50), "New_Age_Category"] = "mature"
df.loc[(df["Age"] >= 50), "New_Age_Category"] = "senior"
df["New_Age_Category"] = df["New_Age_Category"].astype("category")


# df["Pregnancies_Zero"] = df["Pregnancies"].apply(lambda x: 1 if x==0 else 0 )

# yaş ve hamilelik değişkenlerinden yeni değişkenler çıkaralım
df.loc[(df["Age"] < 18 ) & (df["Pregnancies"] != 0 ), "New_AgePregnancies_Category"] = "young_mother"
df.loc[(df["Age"] >= 18) & (df["Age"] < 50) & (df["Pregnancies"] != 0 ), "New_AgePregnancies_Category"] = "mature_mother"
df.loc[(df["Age"] >= 50) & (df["Pregnancies"] != 0 ), "New_AgePregnancies_Category"] = "senior_mother"
df["New_AgePregnancies_Category"] = df["New_AgePregnancies_Category"].astype("category")


# BMI 18,5 aşağısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obez
# df.loc[(df["BMI"] <= 18.5), "New_BMI_Category" ] = "underweight"
# df.loc[(df["BMI"] > 18.5) & (df["BMI"] < 24.9), "New_BMI_Category" ] = "normal"
# df.loc[(df["BMI"] > 24.9) & (df["BMI"] < 29.9), "New_BMI_Category" ] = "Overweight"
# df.loc[(df["BMI"] > 30), "New_BMI_Category" ] = "obez"
df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],labels=["Underweight", "Healthy", "Overweight", "Obese"])

# Glucose değişkeninden yeni değişkenler çıkaralım
df["New_Glucose_Category"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])
# df.loc[(df["Glucose"] < 140), "New_Glucose_Category"] = "Normal"
# df.loc[(df["Glucose"] >=140) & (df["Glucose"] < 200), "New_Glucose_Category"] = "Prediabetes"
# df.loc[(df["Glucose"] > 300), "New_Glucose_Category"] = "Diabetes"

# BMI ve Yaş değişkenlerinden yeni değişkenler çıkaralım
df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"
df["NEW_AGE_BMI_NOM"] = df["NEW_AGE_BMI_NOM"].astype("category")


# yaş ve glucose değişkenlerinden yeni değişkenler oluşturalım
df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"
df["NEW_AGE_GLUCOSE_NOM"] = df["NEW_AGE_GLUCOSE_NOM"].astype("category")

# Adım 3: Encoding işlemlerini gerçekleştiriniz.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]


for col in binary_cols:
    df = label_encoder(df, col)

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)


df.head()

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Adım 5: Model oluşturunuz.

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")
