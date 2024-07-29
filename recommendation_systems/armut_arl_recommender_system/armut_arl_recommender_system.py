# Armut - Association Rule Based Recommender System
# UserId : Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi
# altında koltuk yıkama servisi)Bir ServiceId farklı kategoriler altında bulanabilir ve
# farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si
# 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
from mlxtend.frequent_patterns import apriori, association_rules

# Görev 1: Veriyi Hazırlama
# Adım 1: armut_data.csv dosyasını okutunuz.
df_ = pd.read_csv("projeler/ArmutARL-221116-004247/ArmutARL-221114-234936/armut_data.csv")
df = df_.copy()
df.describe().T
df.isnull().sum()
df.shape

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID’yi "_" ile birleştirerek bu hizmetleri
# temsil edecek yeni bir değişken oluşturunuz.

df["Hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)

# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet
# tanımı (fatura vb. ) bulunmamaktadır. Association Rule Learning uygulayabilmek için bir
# sepet (fatura vb.) tanımı oluşturulması gerekmektedir. Burada sepet tanımı her bir müşterinin
# aylık aldığı hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4
# hizmetleri bir sepeti; 2017’in 10.ayında aldığı 9_4, 38_4 hizmetleri başka bir sepeti
# ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir. Bunun için
# öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz. UserID ve yeni
# oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında yeni bir değişkene atayınız.

df["New_Date"] = df["CreateDate"].apply(pd.to_datetime)
df["New_Date"] = df["New_Date"].dt.strftime('%Y-%m')

df["SepetID"] = df["UserId"].astype(str) + "_" + df["New_Date"]

# Görev 2: Birliktelik Kuralları Üretiniz ve Öneride bulununuz
#Adım 1: Hizmet pivot table’i oluşturunuz.
# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

df_arl = df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
# .unstack(): Hizmet kolonunu sütunlar haline getirir ve her SepetID için bir satır oluşturur. Böylece, SepetID'ye
# göre satırlar ve Hizmet'e göre sütunlar elde edilir. Eğer bir SepetID ve Hizmet kombinasyonu yoksa NaN
# (eksik değer) olur.

# Adım 2: Birliktelik kurallarını oluşturunuz.

apriori_ = apriori(df_arl, min_support=0.01, use_colnames= True)

apriori_.sort_values("support", ascending= False)
rules = association_rules(apriori_, metric= "support", min_threshold= 0.01)
rules.head()

#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

def arl_recommender(rules_df, product_id, rec_count=1):
    #rec_count kaç ürün önereceğini temsil ediyor
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    # recommendation_list'te tekrarlamayı önlemek için aşağıdaki satırı yazıyoruz
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]

arl_recommender(rules, "2_0", 5)
#['15_1', '25_0', '13_11', '38_4', '22_0']








