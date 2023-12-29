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
Dataset yang digunakan adalah dataset yang diambil dari kaggle, dimana isi dari Body Fat Prediction Dataset ini yaitu hasil perkiraan persentase lemak tubuh yang ditentukan di bawah air dengan pengukuran lingkar tubuh 252 pria dan diteliti berdasarkan 14 atribut diatas.

[Body Fat Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset/data). 

### Variabel-variabel pada Body Fat Prediction Dataset adalah sebagai berikut:
- Density : merupakan kepadatan tubuh dalam penimbangan dibawah air.(Float)
- BodyFat : merupakan lemak tubuh dalam persen.(Float)
- Age : merupakan umur dalam satuan tahun.(int)
- Weight : merupakan berat badan dalam satuan pon.(Float)
- Height : merupakan tinggi badan dalam satuan inchi.(Float)
- Neck : merupakan ukuran lingkar leher dalam satuan cm.(Float)
- Chest : merupakan ukuran ukuran lingkar dada dalam satuan cm.(Float)
- Abdomen : merupakan ukuran lingkar perut dalam satuan cm.(Float)
- Hip : merupakan ukuran lingkar panggul dalam satuan cm.(Float)
- Thigh : merupakan ukuran lingkar paha dalam satuan cm.(Float)
- Knee : merupakan ukuran lingkar lutut dalam satuan cm.(Float)
- Ankle : merupakan ukuran lingkar pergelangan kaki dalam satuan cm.(Float)
- Biceps : merupakan ukuran lingkar tangan atas dalam satuan cm.(Float)
- Forearm : merupakan ukuran lingkar tangan bawah dalam satuan cm.(Float)
- Wrist : merupakan ukuran lingkar pergelangan tangan dalam satuan cm.(Float)

## Data Preparation
Data berdasarkan kaggle

Pertama import dulu library yang di butuh dengan memasukan perintah :
```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

Kemudian agar dataset di dalam kaggle langsung bisa terhubung ke collab maka harus membuat token terlebih dahulu di akun kaggle dengan memasukan perintah : 
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
!kaggle datasets download -d fedesoriano/body-fat-prediction-dataset
```

Jika berhasil, selanjutnya kita ekstrak dataset yang sudah didownload dengan perintah :
```bash
!mkdir body-fat-prediction-dataset
!unzip body-fat-prediction-dataset.zip -d body-fat-prediction-dataset
!ls body-fat-prediction-dataset
```

Jika berhasil diekstrak, maka kita langsung dapat membuka dataset tersebut dengan perintah :
```bash
df = pd.read_csv('/content/body-fat-prediction-dataset/bodyfat.csv')
```

Lalu kita dapat melakukan beberapa exploratory daya analysis sederhana, mulai dari menampilkan isi
dari dataset bodyfat.csv dengan memasukan perintah :
```bash
df.head()
```

Jika ingin menampilkan jumlah data dan kolom yang ada di dataset, masukan perintah :
```bash
df.shape
```

Lalu kita cek tipe data dari masing-masing atribut/fitur dari dataset dari bodyfat.csv , masukan perintah :
```bash
df.info()
```

Untuk menampilkan detail informasi dari dataset, masukan perintah :
```bash
df.describe()
```

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


