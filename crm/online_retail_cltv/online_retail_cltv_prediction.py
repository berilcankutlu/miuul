import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", None)
df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.isnull().sum()
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"]>0]
df = df[df["Price"]>0]


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
analysis_date = dt.datetime(2010,12,11)
df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv = df.groupby("Customer ID").agg({"InvoiceDate": [lambda x: (x.max()-x.min()).days,
                                                      lambda date: (analysis_date - date.min()).days],
                                      "Invoice": lambda num: num.nunique(),
                                      "TotalPrice": lambda price: price.sum()})

cltv.columns = ["recency", "T", "frequency", "monetory"]
cltv["monetory"] = cltv["monetory"] / cltv["frequency"]
cltv = cltv[(cltv["recency"] > 1)]
cltv["T"]= cltv["T"] / 7

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv["frequency"], cltv["recency"], cltv["T"])
cltv["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'])
cltv["exp_sales_12_month"] = bgf.predict(4*12,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'])
cltv["exp_sales_1_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'])

