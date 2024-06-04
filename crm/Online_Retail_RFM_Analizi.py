# Görev 1
import pandas as pd
import datetime as dt
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", None)
df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.shape
# betimsel istatistikler
df.describe().T

#boş verilerin çıkarılması
df.isnull().sum()
df.dropna(inplace=True)

# eşsiz ürün sayııs
df["Description"].nunique()

#hangi üründen kaçar tane var
df["Description"].value_counts().head()

# en çok sipariş edilen 5 ürün
df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head(5)

# başında c olanlar iptal olan faturalar
df = df[~df["Invoice"].str.contains("C", na=False)]

# total price oluşturulması
df["TotalPrice"] = df["Quantity"] * df["Price"]

# Görev 2
# r, f, m tanımları
analysis_date = dt.datetime(2010,12,11)
rfm = df.groupby("Customer ID").agg( { "InvoiceDate": lambda date: (analysis_date-date.max()).days, "Invoice": lambda num: num.nunique(), "TotalPrice": lambda price: price.sum()})
rfm.columns = ["recency", "frequency", "monetory"]

# Görev 3
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
rfm["monetory_score"] = pd.qcut(rfm["monetory"],5,labels=[1,2,3,4,5])
rfm["RFM_score"] = (rfm["recency_score"].astype(str)+rfm["frequency_score"].astype(str))

# Görev 4
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

rfm["segments"] = rfm["RFM_score"].replace(seg_map, regex=True)

# Görev 5
target_customer = rfm[rfm["segments"].isin(["loyal_customers"])]
target_customer.to_excel("loyals.xlsx", index=False)
