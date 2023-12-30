# Laporan Proyek Machine Learning
### Nama : Khilda Fitria Nurultsani
### Nim : 211351070
### Kelas : Teknik Informatika Pagi B

## Domain Proyek
Proyek yang saya angkat adalah perhitungan lemak tubuh yang diambil dari 14 parameter yang telah ditentukan. Menurut peneliti, Peningkatan lemak dalam tubuh dapat berpengaruh dalam perubahan bentuk tubuh manusia. Maka dari itu, saya selaku pembuat mencoba membuat pengukur kadar lemak sebagai tindakan agar anda dapat mengetahui jumlah lemak yang ada dalam tubuh dan mencegah terjadinya obesitas.

## Business Understanding
Proyek ini memudahkan serta menghemat waktu kita dalam pengukuran lemak tubuh yang akurat, agar kita dapat mengetahui kadar jumlah lemak yang ada pada tubuh kita tanpa harus pergi ke rumahsakit atau membeli alatnya terlebih dulu.

Bagian laporan ini mencakup :
### Problem Statements
- Ketidaktahuan seseorang terhadap jumlah lemak yang ada dalam tubuhnya.

### Goals
- Untuk mengetahui kadar lemak dalam tubuh, sehingga bisa membantu memantau kondisi kesehatan.

    ### Solution statements
- Dikembangkannya perhitungan lemak tubuh berbasis web agar dapat mengetahui dengan mudah jumlah lemak yang ada dalam tubuh kita dengan parameter yang telah ditentukan dan dihitung menggunakan algoritma Regresi Linear.

## Data Understanding
Dataset yang digunakan adalah dataset yang diambil dari kaggle, Dataset Transactions from a bakery ini terdiri dari 21.293 observasi dari sebuah toko roti. Isinya yaitu seluruh data transaksi konsumen yang berbelanja pada toko roti.

[Transactions from a bakery](https://www.kaggle.com/datasets/sulmansarwar/transactions-from-a-bakery). 

### Variabel-variabel pada Transactions from a bakery adalah sebagai berikut:
- Date : merupakan tanggal transaksi konsumen saat membeli produk di toko roti.
- Time : merupakan waktu transaksi konsumen saat membeli produk di toko roti.
- Transaction : merupakan jumlah produk yang konsumen beli di toko roti. 
- Item : merupakan menu makanan produk yang dijual di toko roti.

## Data preprocessing
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

## Import Library
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

## Import Dataset
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

Jika berhasil diekstrak, maka kita langsung dapat membuka dataset tersebut dengan perintah :
```bash
bakery = pd.read_csv('/content/transactions-from-a-bakery/BreadBasket_DMS.csv')
```

## Data Discovery
Lalu kita dapat melakukan beberapa proses pengumpulan data sederhana, mulai dari menampilkan isi
dari dataset BreadBasket_DMS.csv dengan memasukan perintah :
```bash
bakery.head()
```

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


## Visualisasi Data
Jika ingin mengecek heatmap dari data kita ada yang kosong atau tidak, masukan perintah :
```bash
sns.heatmap(df.isnull())
```
```bash
out :
```
<img width="338" alt="Screenshot 2023-10-26 230013" src="https://github.com/khildafitria/machinelearning/assets/149028314/60ba8229-27ff-4895-a4f2-7f23b5730a3a">

Menggunakan heatmap untuk melihat sebaran data pada dataset bodyfat.csv , masukan perintah :
```bash
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
```
```bash
out :
```
<img width="430" alt="image" src="https://github.com/khildafitria/machinelearning/assets/149028314/9724d627-cf6d-4794-8826-93098ef3690f">


Menampilkan informasi umur berdasarkan berat maka masukan perintah :
```bash
models=df.groupby('Age').count()[['Weight']].sort_values(by='Age', ascending=True).reset_index()
models=models.rename(columns={'Age':'Height'})
```
```bash
fig=plt.figure(figsize=(15,5))
sns.barplot(x=models['Height'], y=models['Weight'], color='orange')
plt.xticks(rotation=60)
```
```bash
out :
```
![image](https://github.com/khildafitria/machinelearning/assets/149028314/9e355699-5971-4123-ab4e-2eede09cdd5e)

Menampilkan distribusi dari fitur lemak tubuh, masukan perintah :
```bash
plt.figure(figsize=(15,5))
sns.histplot(df['BodyFat'])
```
```bash
out :
```
![image](https://github.com/khildafitria/machinelearning/assets/149028314/bec186f1-c2b2-4dda-ad66-1979503ba7cb)

## Modeling
Untuk melakukan modeling saya memakai algoritma regresi linear, dimana kita harus memisahkan mana saja atribut yang akan dijadikan sebagai fitur(x) dan atribut mana yang dijadikan label(y).
```bash
features = ['Density', 'Age', 'Weight', 'Height', 'Neck', 'Chest','Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm','Wrist']
x = df[features]
y = df['BodyFat']
x.shape, y.shape
```
Pada perintah tersebut kita gunakan Density, Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, dan Wrist sebagai fitur inputan(x). Sedangkan BodyFat dijadikan sebagai label(y), karena BodyFat merupakan nilai yang akan diestimasi.


Berikutnya lakukan split data, yaitu memisahkan data training dan data testing dengan memasukan perintah :
```bash
from sklearn.model_selection import train_test_split
x_train, X_test, y_train, y_test = train_test_split(x,y,random_state=70)
y_test.shape
```

Lalu masukan data training dan testing ke dalam model regresi linier dengan perintah :
```bash
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(X_test)
```

Untuk mengecek akurasinya masukan perintah :
```bash
score = lr.score(X_test, y_test)
print('akurasi model regresi linear =', score)
```
```bash
out : akurasi model regresi linear = 0.9909700768437055
```
Didapatkan nilai akurasi 99% , hal ini dipengaruhi oleh jumlah parameter yang digunakan. Jika parameternya dikurangi maka tingkat akurasinya terpengaruh.


Selanjutnya mencoba menggunakan model estimasi menggunakan regresi linier dengan memasukan perintah :
```bash
input_data = np.array([[1.0708 , 23 ,	154.25 ,	67.75 ,	36.2 ,	93.1 ,	85.2 ,	94.5 ,	59.0	, 37.3 ,	21.9 ,	32.0 ,	27.4 ,	17.1]])
prediction = lr.predict(input_data)
print('Perkiraan Lemak Tubuh Dalam Persen :', prediction)
```
```bash
out : Perkiraan Lemak Tubuh Dalam Persen : [11.87458955]
```

Berdasarkan data yang telah diteliti, maka kita dapat mengetahui kadar lemak yang ada dalam tubuh kita.

## Evaluation
Metrik evaluasi yang digunakan yaitu precision dengan memasukan perintah :
```bash
from sklearn.metrics import r2_score
score = r2_score(y_test, pred)

print(f"precision = {score}")
```
```bash
out : precision = 0.9909700768437055
```
Hasil dari metrik evaluasinya 99% sama dengan hasil akurasi yang sebelumnya dilakukan. Berarti model ini memiliki keseimbangan yang baik dari segi presisinya. 
- Saya memilih menggunakan metrik **precision**. Karena dalam mendeteksi lemak tubuh, kesalahan dalam mendeteksi yang sebenarnya tidak ada menyebabkan kecemasan yang tidak perlu atau biaya tambahan untuk tes lebih lanjut.
- Dengan menggunakan presisi membantu dalam mengukur sejauh mana model ini dapat menghindari kesalahan.
  
## Deployment
[Perhitungan Lemak Tubuh](https://machinelearning-khilda.streamlit.app/). 

![image](https://github.com/khildafitria/machinelearning/assets/149028314/ccb44c6a-6415-45fe-af51-dd1c91f0dcea)


