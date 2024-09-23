# Görev 1: Veriyi Hazırlama
# Adım 1: Online Retail II veri setinden 2010-2011 sheet’ini okutunuz.
import pandas as pd
pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
pd.set_option("display.width",500)
from mlxtend.frequent_patterns import apriori,association_rules

df = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011", engine="openpyxl")
df.head()
df.describe().T
df.isnull().sum()
first_user = 21987
second_user = 23235
third_user = 22747
# Adım 2: StockCode’u POST olan gözlem birimlerini drop ediniz.
# (POST her faturaya eklenen bedel, ürünü ifade etmemektedir.)
df = df[~df["StockCode"].str.contains("POST", na=False)]
# Adım 3: Boş değer içeren gözlem birimlerini drop ediniz.
df.dropna(inplace=True)
# Adım 4: Invoice içerisinde C bulunan değerleri veri setinden çıkarınız.
# (C faturanın iptalini ifade etmektedir.)
df = df[~df["Invoice"].str.contains("C", na=False)]
# Adım 5: Price değeri sıfırdan küçük olan gözlem birimlerini filtreleyiniz.
df = df[df["Price"] > 0]
# Adım 6: Price ve Quantity değişkenlerinin aykırı değerlerini inceleyiniz, gerekirse baskılayınız.
df = df[df["Quantity"] > 0]

# Görev 2: Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme
# Adım 1: Aşağıdaki gibi fatura ürün pivot table’i oluşturacak create_invoice_product_df
# fonksiyonunu tanımlayınız.
df_gr = df[df["Country"]=="Germany"]
def create_invoice_product_df(df, id=False):
    if id:
        return df.groupby(["Invoice","StockCode"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x: 1 if x>0 else 0)
    else:
        return df.groupby(["Invoice","Description"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x: 1 if x>0 else 0)


# Adım 2: Kuralları oluşturacak create_rules fonksiyonunu tanımlayınız ve alman müşteriler
# için kurallarını bulunuz.

def create_rules(df, id=True, country="Germany"):
    df = df[df["Country"]== country]
    df = create_invoice_product_df(df,id)
    frequent_itemsets = apriori(df, min_support=0.01, use_colnames= True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold= 0.01)
    return rules

gr_inv_pro_df = create_rules(df_gr)

# Görev 3: Sepet İçerisindeki Ürün Id’leri Verilen Kullanıcılara Ürün Önerisinde Bulunma
# Adım 1: check_id fonksiyonunu kullanarak verilen ürünlerin isimlerini bulunuz.
def check_id(df, stock_code):
    product_name = df[df["StockCode"]== stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df, first_user)
check_id(df, second_user)
check_id(df, third_user)

# Adım 2: arl_recommender fonksiyonunu kullanarak 3 kullanıcı için ürün önerisinde bulununuz.
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0: rec_count]


print(arl_recommender(gr_inv_pro_df, 21987, 2))
# [22467, 21915]
print(arl_recommender(gr_inv_pro_df, 23235, 3))
# [22716, 22467, 22077]
print(arl_recommender(gr_inv_pro_df, 22747, 3))
# [21915, 21915, 22037]

# Adım 3: Önerilecek ürünlerin isimlerine bakınız.
print(check_id(df, 21987))
print(check_id(df, 23235))
print(check_id(df, 22747))




