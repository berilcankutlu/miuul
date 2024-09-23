##########################################################################
# FLO RFM Analizi ile Müşteri Segmentasyonu
##########################################################################
# Görev 1: Veriyi Anlama ve Hazırlama
import pandas as pd
import datetime as dt

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format",lambda x : "%.5f" %x)

df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy() # bir sorun yaşamamak adına kopya alıyorum

# ilk 10 gözlem
df.head(10)
# boyut
df.shape
# değişken isimleri
df.columns
# betimsel istatistik
df.describe().T
# boş değer
df.isnull().sum()
# hiç boş değer olmadığı için silme işlemi yapmayacağız
# değişken tipleri
df.info()
# her bir müşteri için toplam alışveriş sayısı ve harcaması değişkeni oluşturulması
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# değişken tipleri incelenmesi ve tarih değişkeninin tipinin date'e çevrilmesi
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

# alışveriş kanalındaki müşteri sayısının toplam alınan ürün sayısının ve toplam harcamaların dağılımı
df.groupby("order_channel").agg({"master_id" : "sum",
                                 "order_num_total": "sum",
                                 "customer_value_total":"sum"})

# en fazla kazancı getiren ilk 10 müşteri
df.groupby("master_id").agg({"customer_value_total": "sum"}).sort_values("customer_value_total", ascending=False).head(10)
# çözümün önerdiği:
df.sort_values("customer_value_total", ascending=False)[:10]

# en fazla sipariş veren ilk 10 müşteri
df.groupby("master_id").agg({"order_num_total": "sum"}).sort_values("order_num_total", ascending= False).head(10)
# çözümün önerdiği
df.sort_values("order_num_total", ascending=False)[:10]

# veri ön hazırlık sürecinin fonksiyonlaştırılması
def data_prep(df):
    df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
    date_columns = df.columns[df.columns.str.contains("date")]
    df[date_columns] = df[date_columns].apply(pd.to_datetime)
    return df

##########################################################################
# Görev 2 : RFM metriklerinin hesaplanması
# recency, monetary, frequency tanımlarının yapılması
df["last_order_date"].max()
analysis_date = dt.datetime(2021,6,1)

rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).dt.days
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]

#########################################################################
# Görev 3: rf skorunu hesaplaması
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels= [5,4,3,2,1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels= [1,2,3,4,5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels= [1,2,3,4,5])

rfm["rf_score"] = (rfm["recency_score"].astype(str)+rfm["frequency_score"].astype(str))

rfm["rfm_score"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))

################################################################################
# Görev 4 : rf skorunun segment oalrak tanımlanması

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segments"] = rfm["rf_score"].replace(seg_map, regex=True)

#############################################################################
# Görev 5: segmentlerin ortalamalarını gözlemleme
rfm[["segments", "recency", "frequency", "monetary"]].groupby("segments").agg(["mean", "count"])

target_customer = rfm[rfm["segments"].isin(["champions","loyal_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_customer)) &(df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
cust_ids.shape

target_customer = rfm[rfm["segments"].isin(["cant_loose","hibernating","new_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_customer)) & ((df["interested_in_categories_12"].str.contains("ERKEK"))|(df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
cust_ids.to_csv("indirim_hedef_müşteri_ids.csv", index=False)

