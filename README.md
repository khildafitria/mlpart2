# Laporan Proyek Machine Learning
### Nama : Khilda Fitria Nurultsani
### Nim : 211351070
### Kelas : Teknik Informatika Pagi B

## Domain Proyek
Proyek yang saya angkat adalah Platform Analisis Pasar pada toko roti. yang diambil dari 4 parameter yang telah ditentukan, berisi data transaksi mencakup informasi produk yang dibeli oleh pelanggan dalam satu transaksi. Maka dari itu, saya mencoba membuat Platform Analisis Pasar diharap dapat membantu toko roti dalam meningkatkan strategi pemasaran dan menawarkan produk yang lebih menarik bagi pelanggan melalui pola pembelian dan keterkaitan produk.

## Business Understanding
Proyek ini memudahkan toko roti untuk memahami produk yang sering dibeli secara bersamaan, sehingga toko roti dapat menyusun strategi yang lebih efektif dalam penempatan produk, penawaran paket, dan promosi. juga dapat meningkatkan kepuasan pelanggan dengan menyediakan kombinasi produk yang sesuai dengan kebutuhan mereka.

Bagian laporan ini mencakup :
### Problem Statements
- Ketidaktahuan toko roti terhadap Pola Keterkaitan Produk mana yang sering dibeli bersamaan oleh pelanggan.

### Goals
- Untuk mengidentifikasi pola pembelian bersama dan memahami hubungan antarproduk, sehingga toko dapat meningkatkan penjualan melalui penawaran paket atau strategi bundling.

    ### Solution statements
- Dikembangkannya Platform Analisis Pasar berbasis web agar dapat mengetahui dengan mudah produk yang sering dibeli secara bersamaan dengan parameter yang telah ditentukan dan dihitung menggunakan algoritma Apriori.

## Data Understanding
Dataset yang digunakan adalah dataset yang diambil dari kaggle, Dataset Transactions from a bakery ini terdiri dari 21.293 observasi dari sebuah toko roti. Isinya yaitu seluruh data transaksi pelanggan yang berbelanja pada toko roti.

[Transactions from a bakery](https://www.kaggle.com/datasets/sulmansarwar/transactions-from-a-bakery). 

### Variabel-variabel pada Transactions from a bakery adalah sebagai berikut:
- Date : merupakan tanggal transaksi konsumen saat membeli produk di toko roti.
- Time : merupakan waktu transaksi konsumen saat membeli produk di toko roti.
- Transaction : merupakan jumlah produk yang konsumen beli di toko roti. 
- Item : merupakan menu makanan produk yang dijual di toko roti.

## Data preparation
Data berdasarkan kaggle

Pertama import dulu library yang di butuh dengan memasukan perintah :
```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mlxtend.frequent_patterns import association_rules, apriori
import warnings
warnings.filterwarnings('ignore')
```

Kemudian agar dataset di dalam kaggle langsung bisa terhubung ke google collab maka harus membuat token terlebih dahulu di akun kaggle dengan memasukan perintah : 
```bash
from google.colab import files
files.upload()
```
Setelah itu lalu masukan file token.

Berikutnya yaitu membuat direktori dengan memasukan perintah :
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

Setelah itu kita panggil url dataset yang ada di website kaggle untuk didownload langsung ke google colab.
```bash
!kaggle datasets download -d sulmansarwar/transactions-from-a-bakery
```

Jika berhasil, selanjutnya kita ekstrak dataset yang sudah didownload dengan perintah :
```bash
!mkdir transactions-from-a-bakery
!unzip transactions-from-a-bakery -d transactions-from-a-bakery
!ls transactions-from-a-bakery
```

Jika berhasil diekstrak, karena saya mendefinisikan data menjadi bakery maka kita langsung dapat membuka dataset tersebut dengan perintah :
```bash
bakery = pd.read_csv('/content/transactions-from-a-bakery/BreadBasket_DMS.csv')
```
Lalu kita dapat melakukan beberapa proses pengumpulan data sederhana, mulai dari menampilkan isi
dari dataset BreadBasket_DMS.csv dengan memasukan perintah :
```bash
bakery.head()
```
```bash
out :
```
<img width="357" alt="image" src="https://github.com/khildafitria/mlpart2/assets/149028314/175e52b1-6fc3-4433-bdc0-73a4cc29a430">

kita cek tipe data dari masing-masing atribut/fitur dari dataset dari BreadBasket_DMS.csv , masukan perintah :
```bash
bakery.info()
```

Laku Jika ingin menampilkan jumlah data dan kolom yang ada di dataset, masukan perintah :
```bash
bakery.shape
```

Jika ingin mengetahui kolom apa saja yang ada pada dataset, masukan perintah :
```bash
bakery.columns
```

Untuk mengubah semua nilai dalam kolom tersebut menjadi huruf kecil (lowercase), masukan perintah :
```bash
bakery['Item'] = bakery['Item'].apply(lambda item: item.lower())
```

Untuk menghapus spasi (whitespace) di awal dan akhir setiap nilai dalam kolom, masukan perintah :
```bash
bakery['Item'] = bakery['Item'].apply(lambda item: item.strip())
```

Untuk menampilkan kolom transaction dan item, maka masukan perintah :
```bash
bakery = bakery[["Transaction", "Item"]].copy()
bakery.head(10)
```
```bash
out :
```
<img width="248" alt="image" src="https://github.com/khildafitria/mlpart2/assets/149028314/41b90925-4b1e-4351-b71d-60e079992a89">

Untuk menggabungkan kolom date dan time. masukan perintah :
```bash
bakery['Datetime'] = pd.to_datetime(bakery['Date'] + ' ' + bakery['Time'], format='%Y-%m-%d %H:%M:%S')
```

Lalu menghapus kolom date dan time sebelumnya, dan menampilkan kolom baru menjadi Datetime. 
Masukan perintah :
```bash
bakery = bakery.drop(['Date','Time'],axis=1)
bakery.head()
```
```bash
out :
```
<img width="359" alt="image" src="https://github.com/khildafitria/mlpart2/assets/149028314/5a1f5cbf-9d20-4ae6-bd19-fedbc324dd3f">

Untuk mengkonversikan kolom Datetime dalam Data Frame bakery menjadi objek waktu. Masukan perintah :
```bash
bakery['Datetime'] = pd.to_datetime(bakery['Datetime'], format= "%Y-%m-%d %H:%M")
```

Untuk mendapatkan tipe data dari kolom Datetime dalam Data Frame bakery. Masukan perintah :
```bash
bakery["Datetime"].dtype
```

Untuk memecah kolom Datetime ke dalam  bulan, hari, dan jam. Masukan perintah :
```bash
bakery["month"] = bakery['Datetime'].dt.month
bakery["day"] = bakery['Datetime'].dt.day
bakery["hour"] = bakery['Datetime'].dt.hour
bakery.head()
```
```bash
out :
```
<img width="470" alt="image" src="https://github.com/khildafitria/mlpart2/assets/149028314/f12357c4-eaa1-422a-a713-ccc586331b45">


## Exploratory Data Analysis
Jika ingin menampilkan 10 produk yang paling laris pada dataset ini, yaitu masukan perintah :
```bash
top_item = bakery['Item'].value_counts().nlargest(10)
custom_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#c2c2f0', '#ffb3e6', '#c2f0c2', '#ff6666', '#c2f0f0', '#ffcc99']

plt.figure(figsize=(8, 8))
plt.pie(top_item, labels=top_item.index, autopct='%1.1f%%', startangle=90, colors=custom_colors)
plt.title('Top 10 Produk Yang Paling Laris')
plt.show()
```
```bash
out :
```

Disini tampil pie chart yang menampilkan urutan 10 produk makanan yang paling laris dimulai dari coffee, bread, tea, cake, pastry, none, sandwich, medialuna, hot chocolate, dan cookies.

![image](https://github.com/khildafitria/mlpart2/assets/149028314/e2613709-419b-45e4-b53e-301f4977d960)


Lalu, jika ingin menampilkan 10 produk yang kurang laris. Masukan perintah :
```bash
bottom_item = bakery['Item'].value_counts().nsmallest(10)

plt.figure(figsize=(12, 6))
sns.countplot(x='Item', data=bakery, order=bottom_item.index, palette='viridis')
plt.xlabel('Item')
plt.ylabel('Count')
plt.title('10 Produk Yang Kurang Laris')
plt.xticks(size=13, rotation=45)
plt.tight_layout()

plt.show()
```
```bash
out :
```

Disini tampil grafik yang menampilkan frekuensi 10 produk makanan yang kurang laris dimulai dari gift voucher, raw bars, polenta, chicken sand, the bart, adjusment, olum & polenta, bacon, fairy doors, dan hack the stack.

![image](https://github.com/khildafitria/mlpart2/assets/149028314/aab8d367-3f95-4457-b374-b8b8544c9b83)


Jika ingin menampilkan penjualan produk berdasarkan tanggal, maka masukan perintah :
```bash
plt.figure(figsize=(8,5))
sns.countplot(x='day',data=bakery)
plt.title('Penjualan Produk Berdasarkan Tanggal')
plt.show()
```
```bash
out :
```
Grafik menampilkan batang-batang yang mewakili frekuensi penjualan produk pada setiap tanggal (hari). Setiap batang pada sumbu x menunjukkan seberapa sering produk terjual pada hari tertentu.

![image](https://github.com/khildafitria/mlpart2/assets/149028314/26e35b62-db04-4bed-bb19-dac288d56615)


Untuk menampilkan penjualan produk berdasarkan waktu (jam) dalam sehari. Masukan perintah :
```bash
plt.figure(figsize=(8,5))
sns.countplot(x='hour',data=bakery)
plt.title('Penjualan Produk Berdasarkan Waktu')
plt.show()
```
```bash
out :
```
![image](https://github.com/khildafitria/mlpart2/assets/149028314/d64cdd77-ec55-438e-9aec-3c65a8bbdebe)


Untuk menampilkan penjualan produk berdasarkan bulan. Masukan perintah :
```bash
plt.figure(figsize=(8,5))
sns.countplot(x='month',data=bakery)
plt.title('Penjualan Produk Berdasarkan Bulan')
plt.show()
```
```bash
out :
```
![image](https://github.com/khildafitria/mlpart2/assets/149028314/2925ccbf-be00-4b3f-b03f-85f77f4cebf2)

## Modeling
Untuk menghitung berapa kali item yang ada pada dataset tersebut muncul dalam transaksi, masukan perintah :
```bash
item_count = bakery.groupby(["Transaction", "Item"])["Item"].count().reset_index(name="Count")
item_count.head(10)
```
```bash
out :
```
<img width="276" alt="image" src="https://github.com/khildafitria/mlpart2/assets/149028314/26dee5f8-2c5e-416f-ab35-fbff1ac96e4c">


Untuk mendapatkan tabel pivot yang memberikan ringkasan tentang jumlah item yang terjual pada setiap transaksi. Masukan perintah :
```bash
item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
print("ukuran dataset : ", item_count_pivot.shape)
item_count_pivot.head(5)
```
```bash
out :
```
<img width="913" alt="image" src="https://github.com/khildafitria/mlpart2/assets/149028314/7379cc17-d658-4e78-9440-a590eaa063dc">


Untuk mengganti nilai-nilai dalam data bakery, semua nilai yang kurang dari atau sama dengan 0 menjadi 0, dan semua nilai yang lebih besar atau sama dengan 1 menjadi 1. Masukan perintah :
```bash
def encode(x):
    if x <=0:
        return 0
    elif x >= 1:
        return 1

item_count_pivot = item_count_pivot.applymap(encode)
item_count_pivot.head()
```
```bash
out :
```
<img width="914" alt="image" src="https://github.com/khildafitria/mlpart2/assets/149028314/f04b8bc0-3e95-4740-8d12-bea11d1a6d55">


Untuk mencetak informasi tentang ukuran data bakery, termasuk jumlah transaksi dan jumlah item. Masukan perintah :
```bash
print("ukuran dataset : ", item_count_pivot.shape)
print("jumlah transaksi : ", item_count_pivot.shape[0])
print("jumlah item : ", item_count_pivot.shape[1])
```
```bash
out :
```
<img width="240" alt="image" src="https://github.com/khildafitria/mlpart2/assets/149028314/3140e256-4ae9-411b-961b-250c318949c0">


Untuk menemukan kumpulan itemset yang memiliki nilai minimum support 0.01 , masukan perintah :
```bash
support = 0.01
frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)
frequent_items.sort_values("support", ascending=False).head(10)
```
```bash
out :
```
<img width="227" alt="image" src="https://github.com/khildafitria/mlpart2/assets/149028314/daa202f4-e274-4e84-8f82-33d218a1258a">


## Visualisasi Data
Untuk memvisualisasikan hubungan antara nilai support dan confidence dengan scatter plot, masukan perintah :
```bash
plt.scatter(rules['support'], rules['confidence'], alpha=0.5, color='green')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules - Support vs Confidence')
plt.show()
```
```bash
out :
```
Hasilnya menunjukkan setiap titik pada plot mewakili satu aturan asosiasi, di mana sumbu x mewakili nilai support, sumbu y mewakili nilai confidence.

![image](https://github.com/khildafitria/mlpart2/assets/149028314/e876c773-86b4-4470-997c-0b52a6f325cd)


Untuk membuat visualisasi jaringan (network visualization) pada dataset BreadBasket_DMS.csv , masukan perintah :
```bash
# Network visualization for association rules
G = nx.Graph()

for index, row in rules.iterrows():
    G.add_edge(', '.join(row['antecedents']), ', '.join(row['consequents']), weight=row['lift'])

# Set the layout
pos = nx.spring_layout(G)

# Draw the network
nx.draw(G, pos, with_labels=True, font_size=7, node_size=1000, node_color='skyblue', font_color='black', font_weight='bold', edge_color='gray', width=[d['weight'] * 0.1 for u, v, d in G.edges(data=True)])
plt.title('Association Rules Network Visualization')
plt.show()
```
```bash
out :
```
Hasilnya menunjukkan hubungan antara antecedents dan consequents dari aturan asosiasi. Node (simpul) merepresentasikan kumpulan item, dan edge (garis hubungan) antara node merepresentasikan aturan asosiasi. Lebar edge menggambarkan nilai lift.

![image](https://github.com/khildafitria/mlpart2/assets/149028314/bf7bceab-5e55-4353-944c-8c1477d866be)

## Evaluasi
Untuk menampilkan metrik lift dan tingkat kepercayaan, kita dapat mengevaluasi seberapa kuat aturan asosiasi yang dihasilkan oleh model. Masukan perintah :
```bash
metric = "lift"
min_treshold = 1

rules = association_rules(frequent_items, metric=metric, min_threshold=min_treshold)[["antecedents","consequents","support","confidence","lift"]]
rules.sort_values('confidence', ascending=False,inplace=True)
rules.head(15)
```
```bash
out :
```
<img width="470" alt="image" src="https://github.com/khildafitria/mlpart2/assets/149028314/11ecf1b5-877b-4c02-9ed4-8570d9a074fe">

## Deployment
[Transaction from a bakery](https://mlpart2-machinelearning.streamlit.app/). 

![image](https://github.com/khildafitria/mlpart2/assets/149028314/41c6abf5-6554-49e7-95a0-2cad37b065a6)

