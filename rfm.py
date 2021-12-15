import datetime as dt
import pandas as pd
from helpers.helpers import *
from helpers.data_prep import *
from helpers.eda import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("datasets/bank_transactions.csv")
df.head()

# Veri setinin betimsel istatiklerine bakalım.
df.describe().T

# Eksik gözlem
df.isnull().sum()

df[["CustAccountBalance"]].isnull().sum() / df.shape[0]
df[["CustLocation"]].isnull().sum() / df.shape[0]
df[["CustGender"]].isnull().sum() / df.shape[0]
df[["CustGender"]].isnull().sum() / df.shape[0]
# Eksik gözlem bulunan değişkenlerin oranı veri seti içinde çok düşük olduğu için direkt dropna yöntemi ile sileceğiz.

df.dropna(inplace=True)
df.isnull().sum()
# Artık veri setimizde eksik gözlem yok
df.head()

# Veri inceleme
df["CustomerID"].nunique()  # 879358
# Görüldüğü üzere 879358 tane eşşiz müşteri var.

df["CustGender"].value_counts()
# Male-->760978
# Female-->280635

####En çok işlem yapılan 5 konum####
a =df.groupby("CustLocation").agg({"CustLocation": "count"})
a.columns = ["count"]
a.sort_values(by="count", ascending = False).head()
#               count
#CustLocation
#MUMBAI        101997
#NEW DELHI      84143
#BANGALORE      81330
#GURGAON        73594
#DELHI          70549

# GÖrüldüğü üzere en çok işlem yapılan ilk 5 konumu çıkardık.

### En fazla işlem olan ilk 5 konum ###

df.groupby("CustLocation").agg({"TransactionAmount (INR)" : "sum"}).sort_values(by="TransactionAmount (INR)", ascending = False).head()

### En çok işlem yapılan ilk 5 gün ###

a =df.groupby("TransactionDate").agg({"TransactionDate": "count"})
a.columns = ["count"]
a.sort_values(by="count", ascending = False).head()

## Buradada görüldüğü üzere en çok işlem yapılan ilk 5 günü bulduk.

### RFM METRİKLERİNİN OLUŞTURULMASI ###
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
df["TransactionDate"].max()# 2016-12-09

today_date = dt.datetime(2016, 12, 11) # En son işlem tarihinden 2 gün sonrasında bu RFM işlemini yapıyoruz diye düşünüp
# 2 gün sonrasını today date yani bügünün tarihi dedik.

# Şimdi Veri seti içerisinden Recency,Frequency ve Monetary değerlerini alacağız.
rfm = df.groupby('CustomerID').agg({'TransactionDate': lambda date: (today_date - date.max()).days,
                                    'TransactionID': lambda num: num.nunique(),
                                    'TransactionAmount (INR)': lambda pay: pay.sum()})



rfm.columns = ['recency', 'frequency', 'monetary'] # Kolon isimlerini değiştirdik.
rfm.head()

#### RFM Skorlarının oluşturulması ####

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1]) # Burada recency değeri en yeniye 1 en eskiye 5 veriyoruz
# Çünkü daha yeni bir müşteriyi 1 numara ile almak istiyoruz.
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str)) # RFM skorlarını çıkardık ve bunun üstünden müşterileri segmente edeceğiz.

rfm["RFM_SCORE"].head()

# RFM isimlendirmesi
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

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm['segment'].head()

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"]) # Burada segmentlerin betimsel
# istattiklerine bakıp şirketlerin isteğine göre gerekli aksiyomlar alınabilir.
