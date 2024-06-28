# AB Testi ile Bidding Yöntemlerinin Dönüşümünün Karşılaştırılması
# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları reklam sayıları gibi bilgilerin yanı sıra
# buradan gelen kazanç bilgileri yer almaktadır. Kontrol ve Test grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleri
# ab_testing.xlsx excel’inin ayrı sayfalarında yer almaktadır. Kontrol grubuna Maximum Bidding, test grubuna Average
# Bidding uygulanmıştır.

# Impression : Reklam görüntüleme sayısı
# Click      : Görüntülenen reklama tıklama sayısı
# Purchase   : Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning    : Satın alınan ürünler sonrası elde edilen kazanç

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Görev 1: Veriyi Hazırlama ve Analiz Etme
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Adım 1: ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı
# değişkenlere atayınız.
import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)
df1 = pd.read_excel("ABTesti/ab_testing.xlsx", sheet_name="Control Group")
df2 = pd.read_excel("ABTesti/ab_testing.xlsx", sheet_name="Test Group")

# veri üzerinde hata yapılma ihtimaline karşı kopya oluşturuyorum

df_control = df1.copy()
df_test = df2.copy()

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz

def control_the_dfs(dataframe, head = 5):
    print("Shape")
    print(dataframe.shape)
    print("Types")
    print(dataframe.dtypes)
    print(" Head")
    print(dataframe.head(head))
    print("Null Values")
    print(dataframe.isnull().sum())
    print("Quantiles")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 1]).T)

control_the_dfs(df_control)
control_the_dfs(df_test)

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.
df_control["group"] = "control"
df_test["group"] = "test"
df = pd.concat([df_control, df_test], axis=0, ignore_index= True)
df.head(5)
df.tail(5)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Görev 2: A/B Testinin Hipotezinin Tanımlanması
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Adım 1: Hipotezi tanımlayınız.
# H0 : M1 = M2 (kontrol grubu ve test grubu satın alma ort arasında fark yoktur)
# H1 : M1!= M2 (kontrol grubu ve test grubu satın alma ort arasında fark vardır)

# Adım 2: Kontrol ve test grubu için purchase (kazanç) ortalamalarını analiz ediniz.

df.groupby("group").agg({"Purchase": "mean"})

# rastlantı mı?

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Görev 3: Hipotez Testinin Gerçekleştirilmesi
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.

# Normallik Varsayımı
test_stat, pvalue = shapiro(df.loc[df["group"] == "control", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value : 0.5891 dolayısıyla H0 reddedilemez

test_stat, pvalue = shapiro(df.loc[df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value: 0.1541 dolayısıyla H0 reddedilemez

# Varyans Homojenliği
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir
test_stat, pvalue = levene(df.loc[df["group"] == "control", "Purchase"],
                           df.loc[df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value: 0.1083 dolayısıyla H0 reddedilemez

#Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz.
# varsayımlar sağlandığı için parametrik yani bağımsız iki örneklem t testi yapılacak

test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "Purchase"],
                              df.loc[df["group"] == "test", "Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value: 0.3493 dolayısıyla H0 reddedilemez

# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma ortalamaları arasında istatistiki
# olarak anlamlı bir fark olup olmadığını yorumlayınız.

# H0 reddedilemez demek her iki grubun satın alma ortalamaları arasında istatiksel olarak anlamlı fark yoktur demek.

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Görev 4: Sonuçların Analizi
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

# İki gruba uygulanan normallik testi sonucu iki grubunda normal dağılıma uyduğu görülmüştür.
# İki grubunda varyans homojenliği incelendiğinde varyansın homojen olduğu görülmüştür.
# İki varsayım sağlandığından "Bağımsız Örneklem T Testi" uygulanmıştır. Sonucunda p değeri
# 0.05'ten büyük olduğundan H0 hipotezi reddedilememiştir

