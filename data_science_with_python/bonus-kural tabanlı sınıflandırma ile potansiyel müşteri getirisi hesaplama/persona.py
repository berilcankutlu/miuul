import seaborn as sns
import pandas as pd
import numpy as np

#1.1,
df=pd.read_csv("D:\miuul\proje\persona.csv")
df.info()
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
df.head()
#1.2
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

#1.3
df["PRICE"].nunique()
#1.4
df["PRICE"].value_counts()
#1.5
df["COUNTRY"].value_counts()
#1.6
df.groupby("COUNTRY").agg({"PRICE":"sum"})
#1.7
df.groupby("SOURCE").agg({"PRICE":"sum"})
#1.8
df.groupby("COUNTRY").agg({"PRICE":"mean"})
#1.9
df.groupby("SOURCE").agg({"PRICE":"mean"})
df.groupby(["SOURCE", "COUNTRY"]).agg({"PRICE":"mean"})
#2
df.groupby(["COUNTRY","SOURCE", "SEX"]).agg({"PRICE":"mean"})
#3
agg_df = df.groupby(by=["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
agg_df.head()
#4
agg_df = agg_df.reset_index()
agg_df.head()
#5
bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]
agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins, labels=mylabels)
agg_df.head()
#6
agg_df['customer_level_based'] = agg_df[["COUNTRY", "SOURCE", "SEX"]].agg(lambda x: '_'.join(x).upper(), axis=1)
agg_df.head(5)
#7
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.head(30)
agg_df.groupby("SEGMENT").agg({"PRICE": "mean"})
#8
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]
new_user = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]
