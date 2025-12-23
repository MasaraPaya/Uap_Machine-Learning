# Klasifikasi Citra Bunga Daisy, Dandelion, Rose, Sunflower, dan Tulip

## Deskripsi Proyek
Proyek ini merupakan implementasi sistem klasifikasi citra bunga menggunakan pendekatan pembelajaran mesin berbasis **Neural Network** dan **Transfer Learning**. Sistem dirancang untuk mengklasifikasikan citra bunga ke dalam lima kelas, yaitu **Daisy, Dandelion, Rose, Sunflower, dan Tulip**.

Proyek ini dikembangkan sebagai bagian dari **Tugas Ujian Akhir Praktikum (UAP) Machine Learning**, dengan tujuan mengimplementasikan model neural network dasar, model pretrained, serta mengintegrasikan model ke dalam sistem website sederhana menggunakan **Streamlit**.

---

## Dataset
- **Nama Dataset**: Flower Recognition CNN Keras  
- **Sumber**: Kaggle  
  https://www.kaggle.com/code/rajmehra03/flower-recognition-cnn-keras  
- **Jumlah Kelas**: 5  
  - Daisy  
  - Dandelion  
  - Rose  
  - Sunflower  
  - Tulip  
- **Jumlah Data**: 8.455 citra dalam 5 folder  
  (jumlah data awal < 5.000 citra, kemudian ditingkatkan melalui augmentasi data)
- **Preprocessing dan Augmentasi**:
  - Resize citra menjadi 224 × 224
  - Normalisasi dan preprocessing sesuai arsitektur model
  - Augmentasi citra menggunakan rotasi, zoom, dan horizontal flip
  - Pembagian data training dan validation menggunakan `validation_split = 0.2`

---

## Model yang Digunakan
Tiga model dibangun dan dibandingkan sesuai ketentuan UAP:

### 1. CNN (From Scratch)
Model Convolutional Neural Network yang dilatih dari awal tanpa menggunakan bobot pretrained. Model ini digunakan sebagai baseline untuk membandingkan performa dengan model transfer learning.

### 2. MobileNetV2 (Transfer Learning)
Model pretrained MobileNetV2 dengan bobot ImageNet. Model dibangun menggunakan **Functional API**, dengan base model dibekukan dan ditambahkan layer GlobalAveragePooling, Dense, dan Dropout untuk klasifikasi.

### 3. EfficientNetB0 (Transfer Learning)
Model pretrained EfficientNetB0 dengan bobot ImageNet. Arsitektur ini mampu mengekstraksi fitur visual secara lebih efisien dan memberikan performa terbaik pada eksperimen.

---

## Evaluasi Model
Evaluasi dilakukan pada data validation dengan metrik:
- Classification Report (accuracy, precision, recall, f1-score)
- Grafik Loss dan Accuracy
- Confusion Matrix

### Ringkasan Hasil Evaluasi
- **CNN (From Scratch)**  
  Accuracy: 0.75  
  Model baseline mampu mengenali pola dasar citra, namun performanya masih terbatas pada kelas dengan variasi bentuk dan warna tinggi.

- **MobileNetV2**  
  Accuracy: 0.89  
  Transfer learning memberikan peningkatan performa yang signifikan dengan ekstraksi fitur yang lebih stabil.

- **EfficientNetB0**  
  Accuracy: 0.94  
  Model dengan performa terbaik, mampu mengklasifikasikan seluruh kelas bunga dengan sangat baik.

---

## Catatan Penyimpanan Model

Model CNN (From Scratch) tidak disertakan di repository GitHub karena ukuran file model yang cukup besar sehingga melebihi batas unggah GitHub. 

Namun demikian, model CNN tetap:
- dilatih dan dievaluasi secara penuh,
- digunakan sebagai baseline dalam analisis,
- dan hasil evaluasinya disertakan dalam tabel perbandingan model.

Model MobileNetV2 dan EfficientNetB0 disertakan dalam repository karena memiliki ukuran file yang lebih efisien dan digunakan pada implementasi website Streamlit.

---

## Tabel Analisis Perbandingan Model

| Nama Model | Akurasi | Hasil Analisis |
|------------|---------|----------------|
| CNN (From Scratch) | 0.75 | Model baseline yang dilatih dari awal tanpa bobot pretrained. Performa cukup baik, namun masih kesulitan pada kelas dengan variasi visual tinggi. |
| MobileNetV2 (Transfer Learning) | 0.89 | Model pretrained dengan transfer learning yang meningkatkan akurasi secara signifikan dan memberikan hasil klasifikasi yang lebih stabil. |
| EfficientNetB0 (Transfer Learning) | 0.94 | Model dengan performa terbaik. Arsitektur EfficientNet mampu mengekstraksi fitur visual secara lebih efektif. |

---

## Implementasi Website (Streamlit)
Sistem website dibangun menggunakan **Streamlit** dan dijalankan secara lokal. Website menyediakan fitur:
- Upload citra bunga
- Pemilihan model (CNN, MobileNetV2, EfficientNetB0)
- Menampilkan hasil prediksi dan confidence score
- Menampilkan probabilitas tiap kelas

---

## Cara Menjalankan Aplikasi
Pastikan seluruh dependensi telah terinstal.

```bash
streamlit run app.py
```


