import seaborn as sns
import pandas as pd
import numpy as np

#1.1
df = pd.read_excel("D:\gezinomi\miuul_gezinomi.xlsx")
df.info()
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
df.head()

#1.2
df["SaleCityName"].unique()
df["SaleCityName"].nunique()

#1.3
df["ConceptName"].unique()
df["ConceptName"].nunique()

#1.4
df.groupby("ConceptName").agg({"SaleCheckInDayDiff": "sum"})

#1.5
df.groupby("SaleCityName").agg({"Price": "sum"})

#1.6
df.groupby("ConceptName").agg({"Price": "sum"})

#1.7
df.groupby("SaleCityName").agg({"Price": "mean"})

#1.8
df.groupby("ConceptName").agg({"Price": "mean"})

#1.9
df.groupby(["ConceptName","SaleCityName"]).agg({"Price": "mean"})

#2
df["SaleCheckInDayDiff"].dtype
df["SaleCheckInDayDiff"].astype(int)
df["SaleCheckInDayDiff"].dtype

bins = [-1, 7, 30, 90, df["SaleCheckInDayDiff"].max()]
labels = ["Last Minuters", "Potential Planners", "Planners", "Early Bookers"]

df["EB_Score"] = pd.cut(df["SaleCheckInDayDiff"], bins, labels=labels)
df.head(50).to_excel("eb_scorew.xlsx", index=False)
df.head()

#3

df.groupby(by=["SaleCityName", "ConceptName","EB_Score"])["Price"].mean()
#or
df.groupby(by=["SaleCityName", "ConceptName","EB_Score"]).agg({"Price":"mean"})
df.groupby(by=["SaleCityName", "ConceptName", "Seasons"]).agg({"Price":"mean"})
df.groupby(by=["SaleCityName", "ConceptName", "CInDay"]).agg({"Price":"mean"})

df.groupby(by=["SaleCityName", "ConceptName","EB_Score"]).agg({"Price":["mean","count"]})
df.groupby(by=["SaleCityName", "ConceptName", "Seasons"]).agg({"Price":["mean","count"]})
df.groupby(by=["SaleCityName", "ConceptName", "CInDay"]).agg({"Price":["mean","count"]})

#4
agg_df=df.groupby(by=["SaleCityName", "ConceptName", "Seasons"]).agg({"Price":"mean"}).sort_values("Price", ascending=False)
agg_df.head(5)

#5
agg_df.reset_index(inplace=True)
agg_df.head(5)

#6
agg_df['sales_level_based'] = agg_df[["SaleCityName", "ConceptName", "Seasons"]].agg(lambda x: '_'.join(x).upper(), axis=1)
agg_df.head(5)

#7
agg_df["SEGMENT"] = pd.qcut(agg_df["Price"], 4, labels=["D", "C", "B", "A"])
agg_df.groupby("SEGMENT").agg({"Price": ["mean", "max", "sum"]})
agg_df.head(5)

#8
agg_df.sort_values(by="Price")


new_user = "ANTALYA_HERŞEY DAHIL_HIGH"
agg_df[agg_df["sales_level_based"] == new_user]

