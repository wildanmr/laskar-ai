# Predictive Analytics Project: PM2.5 Level Prediction in Beijing

## 1. Domain Proyek

### Latar Belakang

Polusi udara, khususnya partikel PM2.5, telah menjadi masalah serius di banyak kota besar di seluruh dunia, termasuk Beijing. Tingkat PM2.5 yang tinggi dapat berdampak negatif pada kesehatan manusia dan lingkungan. Oleh karena itu, kemampuan untuk memprediksi kadar PM2.5 sangat penting untuk memungkinkan pihak berwenang dan masyarakat mengambil tindakan pencegahan yang tepat.

### Permasalahan

Proyek ini bertujuan untuk mengatasi tiga permasalahan utama:

1.  **Faktor-faktor apa saja yang paling berpengaruh terhadap tingkat polusi PM2.5 di Beijing?**
    *   Analisis ini akan membantu mengidentifikasi variabel-variabel kunci (misalnya, kondisi cuaca, polutan lain) yang memiliki korelasi kuat dengan konsentrasi PM2.5.

2.  **Bagaimana mengembangkan model machine learning untuk memprediksi kadar PM2.5 berdasarkan data cuaca dan polutan lainnya?**
    *   Fokusnya adalah membangun model prediktif yang akurat menggunakan teknik regresi.

3.  **Model prediksi mana yang memberikan hasil terbaik untuk estimasi PM2.5 — Regresi Linear atau Random Forest?**
    *   Perbandingan kinerja antara model-model ini akan dilakukan untuk menentukan pendekatan yang paling efektif.

## 2. Business Understanding

### Problem Statements

*   Tingginya kadar PM2.5 di Beijing memerlukan sistem prediksi yang akurat untuk mitigasi dampak kesehatan dan lingkungan.
*   Kurangnya pemahaman mendalam tentang faktor-faktor pendorong utama polusi PM2.5 menghambat pengembangan strategi pengendalian yang efektif.

### Goals

*   Mengidentifikasi faktor-faktor lingkungan dan meteorologi yang paling signifikan yang mempengaruhi konsentrasi PM2.5.
*   Mengembangkan model machine learning yang mampu memprediksi kadar PM2.5 dengan akurasi tinggi.
*   Membandingkan kinerja model Regresi Linear dan Random Forest untuk menentukan model terbaik untuk prediksi PM2.5.

### Solution Statement

Untuk mencapai tujuan ini, kami akan menerapkan pendekatan machine learning dengan membandingkan beberapa algoritma regresi. Secara spesifik, kami akan:

1.  Menggunakan model **Regresi Linear** sebagai baseline untuk memahami hubungan linier antara fitur dan target.
2.  Menggunakan model **Random Forest Regressor** yang dikenal mampu menangani hubungan non-linier dan interaksi fitur yang kompleks.

Kinerja model akan diukur menggunakan metrik seperti Root Mean Squared Error (RMSE) dan R-squared (R2 Score) untuk memastikan solusi yang terukur dan dapat diandalkan.

## 3. Data Understanding

Dataset yang digunakan dalam proyek ini adalah "Beijing Multisite Air Quality Data" yang berisi data kualitas udara dan meteorologi dari berbagai stasiun di Beijing. Dataset ini mencakup observasi per jam dari tahun 2013 hingga 2017.

### Informasi Dataset

*   **Jumlah Sampel:** 420,768 entri.
*   **Kolom:** 17 kolom, termasuk informasi waktu (tahun, bulan, hari, jam), konsentrasi polutan (PM2.5, PM10, SO2, NO2, CO, O3), data meteorologi (TEMP, PRES, DEWP, RAIN, wd, WSPM), dan nama stasiun.
*   **Tipe Data:** Campuran `int64`, `float64`, dan `object` (untuk `wd` dan `station`).

### Deskripsi Kolom Penting:

*   **PM2.5:** Konsentrasi partikel PM2.5 (target variabel).
*   **PM10:** Konsentrasi partikel PM10.
*   **SO2, NO2, CO, O3:** Konsentrasi polutan gas.
*   **TEMP:** Suhu (Celsius).
*   **PRES:** Tekanan atmosfer (hPa).
*   **DEWP:** Titik embun (Celsius).
*   **RAIN:** Curah hujan (mm).
*   **wd:** Arah angin.
*   **WSPM:** Kecepatan angin (m/s).
*   **station:** Nama stasiun pemantauan.

### Missing Values

Selama tahap eksplorasi awal, ditemukan beberapa kolom memiliki nilai yang hilang (NaN):

*   `PM2.5`: 8739 missing values
*   `PM10`: 6449 missing values
*   `SO2`: 9021 missing values
*   `NO2`: 12116 missing values
*   `CO`: 20701 missing values
*   `O3`: 13277 missing values
*   `TEMP`: 398 missing values
*   `PRES`: 393 missing values
*   `DEWP`: 403 missing values
*   `RAIN`: 390 missing values
*   `wd`: 1822 missing values
*   `WSPM`: 318 missing values

Kolom `year`, `month`, `day`, `hour`, dan `station` tidak memiliki nilai yang hilang.

### Unique Values

*   **station:** Terdapat 12 stasiun pemantauan yang unik di Beijing.
*   **wd:** Terdapat 16 arah angin yang unik, ditambah nilai `nan` yang menunjukkan missing values.

Observasi awal menunjukkan bahwa dataset ini cukup komprehensif untuk tujuan prediksi PM2.5, namun memerlukan penanganan missing values dan rekayasa fitur untuk kolom waktu dan kategorikal.

## 4. Data Preparation

Tahap persiapan data sangat krusial untuk memastikan kualitas data yang masuk ke model machine learning. Langkah-langkah berikut telah dilakukan:

### 4.1 Penanganan Missing Values

*   **Kolom Numerik:** Untuk kolom numerik yang memiliki nilai hilang (PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM), nilai-nilai yang hilang diisi menggunakan nilai median dari masing-masing kolom. Pendekatan ini dipilih karena median kurang sensitif terhadap outlier dibandingkan mean.
*   **Kolom Kategorikal (`wd`):** Untuk kolom arah angin (`wd`), nilai yang hilang diisi menggunakan modus (nilai yang paling sering muncul) dari kolom tersebut.

### 4.2 Pembuatan Fitur Berbasis Waktu

Dari kolom `year`, `month`, `day`, dan `hour`, dibuat fitur-fitur berbasis waktu baru untuk menangkap pola musiman dan tren:

*   `date`: Kombinasi `year`, `month`, `day`, dan `hour` menjadi objek datetime.
*   `day_of_week`: Hari dalam seminggu (0=Senin, 6=Minggu).
*   `day_of_year`: Hari dalam setahun.
*   `week_of_year`: Minggu dalam setahun.
*   `quarter`: Kuartal dalam setahun.

Kolom `year`, `month`, `day`, `hour`, dan `date` asli kemudian dihapus karena informasi telah diekstraksi ke fitur baru.

### 4.3 Encoding Fitur Kategorikal

*   **`wd` (Arah Angin):** Karena arah angin memiliki urutan alami (sirkular), `Label Encoding` digunakan untuk mengubahnya menjadi representasi numerik (`wd_encoded`).
*   **`station` (Stasiun):** Karena tidak ada urutan alami antar stasiun, `One-Hot Encoding` diterapkan pada kolom `station`. Ini menghasilkan kolom biner baru untuk setiap stasiun, menunjukkan keberadaan observasi di stasiun tersebut.

### 4.4 Feature Scaling

Fitur-fitur numerik (`PM10`, `SO2`, `NO2`, `CO`, `O3`, `TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM`, `day_of_year`, `week_of_year`) diskalakan menggunakan `StandardScaler`. Scaling ini penting untuk model berbasis jarak dan gradien, memastikan semua fitur berkontribusi secara proporsional dan mencegah fitur dengan skala besar mendominasi proses pelatihan.

### 4.5 Pembagian Data

Dataset dibagi menjadi tiga subset:

*   **Data Latih (Training Set):** 60% dari data, digunakan untuk melatih model.
*   **Data Validasi (Validation Set):** 20% dari data, digunakan untuk menyetel hyperparameter dan mencegah overfitting selama pelatihan.
*   **Data Uji (Test Set):** 20% dari data, digunakan untuk evaluasi akhir kinerja model yang tidak terlihat selama pelatihan atau penyetelan.

Pembagian ini dilakukan secara acak dengan `random_state=42` untuk memastikan reproduktifitas.

## 5. Modeling

Pada tahap ini, tiga jenis model machine learning diimplementasikan untuk memprediksi kadar PM2.5: Regresi Linear, Random Forest Regressor, dan Long Short-Term Memory (LSTM).

### 5.1 Linear Regression

*   **Implementasi:** Model Regresi Linear adalah model statistik dasar yang memodelkan hubungan antara variabel dependen (PM2.5) dan satu atau lebih variabel independen (fitur) sebagai fungsi linier. Model ini berfungsi sebagai baseline untuk membandingkan kinerja model yang lebih kompleks.
*   **Pelatihan:** Model dilatih menggunakan data latih yang telah diskalakan.
*   **Evaluasi:** Kinerja model dievaluasi pada data validasi menggunakan Root Mean Squared Error (RMSE) dan R-squared (R2 Score).

### 5.2 Random Forest Regressor

*   **Implementasi:** Random Forest adalah algoritma ensemble yang membangun banyak pohon keputusan selama pelatihan dan menghasilkan output yang merupakan rata-rata prediksi dari masing-masing pohon. Model ini dikenal karena kemampuannya menangani hubungan non-linier, interaksi fitur yang kompleks, dan mengurangi overfitting.
*   **Pelatihan:** Model dilatih menggunakan data latih yang telah diskalakan. Untuk demonstrasi awal, `n_estimators` diatur ke 100, namun dalam skenario produksi, nilai ini dapat ditingkatkan dan hyperparameter lainnya dapat disetel untuk kinerja optimal.
*   **Evaluasi:** Kinerja model dievaluasi pada data validasi menggunakan RMSE dan R2 Score.

## 6. Evaluation

Evaluasi model dilakukan berdasarkan metrik Root Mean Squared Error (RMSE) dan R-squared (R2 Score) pada data validasi. RMSE mengukur rata-rata besarnya kesalahan prediksi model, di mana nilai yang lebih rendah menunjukkan kinerja yang lebih baik. R2 Score menunjukkan proporsi varians dalam variabel dependen yang dapat dijelaskan oleh model, di mana nilai yang lebih tinggi menunjukkan kecocokan model yang lebih baik.

### Hasil Evaluasi pada Data Validasi:

**Linear Regression:**
*   RMSE: 31.6726
*   R2 Score: 0.8453

**Random Forest:**
*   RMSE: 18.0846
*   R2 Score: 0.9496

### Perbandingan dan Pemilihan Model:

Dari hasil di atas, terlihat jelas bahwa model **Random Forest** menunjukkan kinerja yang jauh lebih unggul dibandingkan dengan model Regresi Linear. RMSE Random Forest (18.0846) secara signifikan lebih rendah daripada RMSE Regresi Linear (31.6726), menunjukkan bahwa prediksi Random Forest memiliki kesalahan rata-rata yang lebih kecil. Selain itu, R2 Score Random Forest (0.9496) jauh lebih tinggi daripada Regresi Linear (0.8453), yang berarti model Random Forest mampu menjelaskan hampir 95% variabilitas dalam kadar PM2.5, dibandingkan dengan sekitar 84.5% oleh Regresi Linear.

Berdasarkan perbandingan ini, **model Random Forest dipilih sebagai model terbaik** untuk memprediksi kadar PM2.5 di Beijing di antara model-model yang berhasil dilatih.


