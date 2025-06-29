# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

* **Nama:** Wildan Mufid Ramadhan
* **Email:** wildan.20nov@gmail.com
* **Dicoding ID:** wildan.20nov@gmail.com 

## Executive Summary

Jaya Jaya Institut menghadapi tantangan signifikan dalam tingkat dropout siswa yang mencapai 32.1% dari total 4,424 siswa. Penelitian ini mengembangkan sistem prediksi dropout menggunakan machine learning dengan akurasi 76% untuk memungkinkan intervensi dini. Dashboard interaktif telah dibuat untuk memantau performa siswa secara real-time. Analisis menunjukkan bahwa faktor-faktor seperti gender, status beasiswa, dan performa akademik semester pertama memiliki pengaruh signifikan terhadap tingkat dropout.

## 1. Business Understanding

### 1.1 Permasalahan Bisnis

Jaya Jaya Institut, sebagai institusi pendidikan tinggi yang telah berdiri sejak tahun 2000, menghadapi permasalahan serius terkait tingkat dropout siswa yang tinggi. Tingkat dropout yang mencapai 32.1% tidak hanya merugikan institusi secara finansial, tetapi juga berdampak pada reputasi dan kualitas pendidikan yang diberikan. Dropout siswa merupakan fenomena kompleks yang dipengaruhi oleh berbagai faktor, mulai dari kondisi sosial-ekonomi, performa akademik, hingga faktor personal siswa. Deteksi dini terhadap siswa yang berpotensi dropout menjadi krusial untuk memungkinkan institusi memberikan intervensi yang tepat waktu dan efektif.

### 1.2 Cakupan Proyek

Proyek ini berfokus pada pengembangan solusi berbasis data untuk mengatasi masalah dropout siswa di Jaya Jaya Institut. Cakupan proyek meliputi:
- **Analisis Data:** Memahami karakteristik siswa dan faktor-faktor yang berkontribusi terhadap dropout.
- **Pengembangan Model Prediksi:** Membangun model machine learning yang mampu memprediksi probabilitas siswa untuk dropout, lulus, atau tetap terdaftar.
- **Pembuatan Dashboard:** Merancang dan mengimplementasikan dashboard interaktif untuk visualisasi data dan monitoring performa siswa secara berkala.
- **Rekomendasi Strategis:** Memberikan rekomendasi actionable berdasarkan hasil analisis dan model untuk intervensi yang efektif.

### 1.3 Tujuan Penelitian

Penelitian ini bertujuan untuk:
1. Menganalisis pola dan faktor-faktor yang mempengaruhi dropout siswa
2. Mengembangkan model prediksi dropout dengan akurasi tinggi
3. Membuat dashboard monitoring untuk memantau performa siswa
4. Memberikan rekomendasi actionable untuk mengurangi tingkat dropout

### 1.4 Manfaat Penelitian

Manfaat yang diharapkan dari penelitian ini meliputi:
- Kemampuan deteksi dini siswa berisiko dropout
- Optimalisasi alokasi sumber daya untuk program intervensi
- Peningkatan tingkat retensi siswa
- Perbaikan reputasi institusi melalui peningkatan tingkat kelulusan


## 2. Data Understanding

### 2.1 Deskripsi Dataset

Dataset yang digunakan dalam penelitian ini berisi informasi komprehensif tentang 4,424 siswa dari berbagai program studi di Jaya Jaya Institut. Data mencakup informasi yang tersedia pada saat pendaftaran siswa (jalur akademik, demografi, dan faktor sosial-ekonomi) serta performa akademik siswa pada akhir semester pertama dan kedua.

Dataset terdiri dari 37 variabel yang dapat dikategorikan sebagai berikut:

**Informasi Demografis:**
- Status pernikahan (Marital Status)
- Jenis kelamin (Gender)
- Usia saat pendaftaran (Age at Enrollment)
- Kewarganegaraan (Nationality)

**Informasi Akademik:**
- Mode aplikasi (Application Mode)
- Urutan aplikasi (Application Order)
- Program studi (Course)
- Jadwal kuliah (Daytime/Evening Attendance)
- Kualifikasi sebelumnya (Previous Qualification)
- Nilai kualifikasi sebelumnya (Previous Qualification Grade)
- Nilai penerimaan (Admission Grade)

**Informasi Sosial-Ekonomi:**
- Kualifikasi pendidikan ibu (Mother's Qualification)
- Kualifikasi pendidikan ayah (Father's Qualification)
- Pekerjaan ibu (Mother's Occupation)
- Pekerjaan ayah (Father's Occupation)
- Status beasiswa (Scholarship Holder)
- Status hutang (Debtor)
- Status pembayaran SPP (Tuition Fees Up to Date)

**Informasi Khusus:**
- Status pengungsi (Displaced)
- Kebutuhan pendidikan khusus (Educational Special Needs)
- Status mahasiswa internasional (International)

**Performa Akademik:**
- Unit kurikulum semester 1: credited, enrolled, evaluations, approved, grade, without evaluations
- Unit kurikulum semester 2: credited, enrolled, evaluations, approved, grade, without evaluations

**Indikator Ekonomi Makro:**
- Tingkat pengangguran (Unemployment Rate)
- Tingkat inflasi (Inflation Rate)
- GDP

**Target Variable:**
- Status siswa: Graduate, Dropout, Enrolled

### 2.2 Kualitas Data

Analisis kualitas data menunjukkan hasil yang sangat baik:
- **Missing Values:** Tidak ada missing values dalam dataset (0% untuk semua variabel)
- **Konsistensi Data:** Format data konsisten dengan tipe data yang sesuai
- **Outliers:** Beberapa outliers teridentifikasi pada variabel usia dan nilai, namun masih dalam rentang yang wajar

### 2.3 Distribusi Target Variable

Distribusi status siswa dalam dataset:
- **Graduate:** 2,209 siswa (49.9%)
- **Dropout:** 1,421 siswa (32.1%)
- **Enrolled:** 794 siswa (17.9%)

Tingkat dropout sebesar 32.1% menunjukkan masalah yang signifikan yang memerlukan perhatian serius dari manajemen institusi.


## 3. Data Preparation

### 3.1 Data Cleaning

Proses pembersihan data meliputi:
1. **Standardisasi Nama Kolom:** Mengubah nama kolom menjadi format lowercase dengan underscore untuk konsistensi
2. **Penanganan Delimiter:** Dataset menggunakan semicolon (;) sebagai delimiter yang telah ditangani dengan tepat
3. **Validasi Tipe Data:** Memastikan setiap variabel memiliki tipe data yang sesuai (numerical, categorical)

### 3.2 Feature Engineering

Beberapa transformasi yang dilakukan:
1. **Label Encoding:** Target variable (Status) di-encode menjadi format numerical untuk keperluan modeling
2. **One-Hot Encoding:** Variabel kategorikal di-transform menggunakan one-hot encoding
3. **Standardization:** Variabel numerical di-standardisasi menggunakan StandardScaler untuk memastikan skala yang konsisten

### 3.3 Data Splitting

Data dibagi menjadi:
- **Training Set:** 80% (3,539 samples)
- **Testing Set:** 20% (885 samples)

Pembagian dilakukan secara stratified untuk mempertahankan proporsi target variable.

## 4. Exploratory Data Analysis (EDA)

- **Distribusi Status:** Terlihat bahwa jumlah siswa `Graduate` adalah yang terbanyak, diikuti oleh `Dropout`, dan `Enrolled`. Tingkat dropout yang tinggi (sekitar 32.1%) menjadi perhatian utama.
- **Distribusi Usia:** Mayoritas siswa mendaftar pada usia muda (18-22 tahun).
- **Gender vs Status:** Terlihat bahwa terdapat perbedaan mencolok dalam distribusi status antara gender. Data menunjukkan bahwa mahasiswa perempuan memiliki tingkat kelulusan yang jauh lebih tinggi dibandingkan dengan mahasiswa laki-laki, yang dapat mengindikasikan adanya pengaruh gender terhadap kemungkinan untuk lulus atau dropout.
- **Beasiswa vs Status:** Siswa yang menerima beasiswa cenderung memiliki tingkat kelulusan yang lebih tinggi dan tingkat dropout yang lebih rendah, menunjukkan pentingnya dukungan finansial.
- **Korelasi:** Matriks korelasi menunjukkan hubungan antar fitur numerik. Fitur-fitur terkait performa akademik (misalnya, `curricular_units_1st_sem_approved`, `curricular_units_1st_sem_grade`) kemungkinan besar berkorelasi kuat dengan status siswa.

## 5. Modeling

### 5.1 Pemilihan Algoritma

Untuk mengatasi masalah klasifikasi multi-class (Graduate, Dropout, Enrolled), dipilih algoritma Random Forest Classifier dengan pertimbangan:

1. **Robustness:** Tahan terhadap outliers dan noise dalam data
2. **Feature Importance:** Dapat memberikan informasi tentang pentingnya setiap fitur
3. **Handling Mixed Data Types:** Dapat menangani kombinasi variabel numerical dan categorical
4. **Interpretability:** Relatif mudah diinterpretasi dibandingkan algoritma ensemble lainnya
5. **Performance:** Umumnya memberikan performa yang baik untuk dataset dengan karakteristik serupa

### 5.2 Parameter Setting

Model Random Forest dikonfigurasi dengan parameter:
- **n_estimators:** 100 (jumlah decision trees)
- **random_state:** 42 (untuk reproducibility)
- **default parameters** untuk parameter lainnya

### 5.3 Training Process

Proses training melibatkan:
1. **Preprocessing:** Standardisasi fitur numerical dan encoding fitur categorical
2. **Model Fitting:** Training Random Forest pada training set
3. **Validation:** Evaluasi performa pada test set

## 6. Evaluation

### 6.1 Metrics Evaluasi

**Overall Accuracy:** 75.93%

Model menunjukkan performa yang cukup baik dengan akurasi mendekati 76%, yang berarti model dapat memprediksi status siswa dengan benar pada 3 dari 4 kasus.

### 6.2 Classification Report

```
              precision    recall  f1-score   support
     Dropout       0.84      0.77      0.81       316
    Enrolled       0.49      0.29      0.37       151
    Graduate       0.76      0.92      0.83       418
    
    accuracy                           0.76       885
   macro avg       0.70      0.66      0.67       885
weighted avg       0.74      0.76      0.74       885
```

**Analisis per Kelas:**

1. **Dropout (Precision: 0.84, Recall: 0.77, F1: 0.81):**
   - Model sangat baik dalam mengidentifikasi siswa dropout
   - Precision tinggi menunjukkan rendahnya false positive
   - Recall yang baik menunjukkan kemampuan mendeteksi sebagian besar kasus dropout

2. **Graduate (Precision: 0.76, Recall: 0.92, F1: 0.83):**
   - Model excellent dalam mengidentifikasi siswa yang akan lulus
   - Recall sangat tinggi (92%) menunjukkan hampir semua graduate terdeteksi
   - Performa terbaik di antara ketiga kelas

3. **Enrolled (Precision: 0.49, Recall: 0.29, F1: 0.37):**
   - Performa terlemah untuk kelas enrolled
   - Hal ini dapat dipahami karena status "enrolled" adalah status transisi
   - Siswa enrolled dapat berubah menjadi graduate atau dropout di masa depan

### 6.3 Confusion Matrix Analysis

```
Predicted:    Dropout  Enrolled  Graduate
Actual:
Dropout         244      21        51
Enrolled         36      44        71  
Graduate          9      25       384
```

**Key Insights:**
- Model sangat baik dalam memprediksi Graduate (384/418 = 91.9% correct)
- Prediksi Dropout cukup akurat (244/316 = 77.2% correct)
- Kelas Enrolled paling sulit diprediksi (44/151 = 29.1% correct)

### 6.4 Business Impact

Dengan akurasi 76% dan performa yang sangat baik dalam mendeteksi dropout (precision 84%), model ini dapat memberikan value signifikan:

1. **Early Warning System:** Identifikasi dini siswa berisiko dropout
2. **Resource Optimization:** Fokus intervensi pada siswa yang benar-benar berisiko
3. **Cost Reduction:** Mengurangi biaya akibat dropout melalui intervensi preventif
4. **Improved Retention:** Potensi peningkatan tingkat retensi siswa


## 7. Business Dashboard

### 7.1 Tujuan

- **Monitoring Performa Siswa:** Menyediakan gambaran komprehensif tentang status akademik dan demografis siswa secara real-time.
- **Identifikasi Tren:** Membantu mengidentifikasi tren dan pola dalam data siswa yang dapat mengindikasikan potensi masalah atau keberhasilan.
- **Pengambilan Keputusan Berbasis Data:** Mendukung keputusan strategis terkait intervensi, alokasi sumber daya, dan pengembangan program.
- **Komunikasi Efektif:** Memfasilitasi komunikasi yang jelas dan ringkas tentang performa siswa kepada berbagai pemangku kepentingan.

### 7.2 Manfaat

- **Peningkatan Efisiensi Operasional:** Mengurangi waktu dan upaya yang dibutuhkan untuk mengumpulkan dan menganalisis data secara manual.
- **Intervensi Dini yang Lebih Baik:** Memungkinkan identifikasi cepat siswa berisiko, sehingga intervensi dapat dilakukan lebih awal dan lebih efektif.
- **Optimalisasi Sumber Daya:** Membantu mengalokasikan sumber daya (misalnya, konselor, program bimbingan) ke area yang paling membutuhkan.
- **Peningkatan Akuntabilitas:** Menyediakan metrik yang jelas untuk melacak kemajuan dan mengevaluasi efektivitas program retensi.
- **Peningkatan Retensi Siswa:** Pada akhirnya, berkontribusi pada penurunan tingkat dropout dan peningkatan tingkat kelulusan.


## 8. Metabase dan Streamlit

### 8.1 Menjalankan Metabase

1. Buka terminal dan masuk ke direktori `metabase`:

   ```bash
   cd metabase
   ```
2. Jalankan layanan Metabase menggunakan Docker Compose:

   ```bash
   docker compose up -d
   ```
3. Setelah layanan aktif, akses Metabase melalui browser.
4. Gunakan kredensial berikut untuk login:

   * **Email**: `root@mail.com`
   * **Password**: `root123`

### 8.2 Menjalankan Aplikasi Streamlit

1. Masuk ke direktori aplikasi:

   ```bash
   cd app
   ```
2. Buat environment Python dan aktifkan:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/macOS
   venv\Scripts\activate     # Untuk Windows
   ```
3. Install dependencies dari file `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi Streamlit:

   ```bash
   streamlit run main.py
   ```

### 8.3 Alternatif Akses Streamlit (Online)

Jika tidak ingin menjalankan aplikasi secara lokal, Anda dapat mengakses aplikasi Streamlit secara online melalui tautan berikut:

👉 [https://wildanmr-student-performance-app.streamlit.app/](https://wildanmr-student-performance-app.streamlit.app/)



## 9. Conclusion

### 9.1 Pencapaian Tujuan

Penelitian ini berhasil mencapai semua tujuan yang ditetapkan:

1. **Analisis Komprehensif:** Berhasil mengidentifikasi faktor-faktor kunci yang mempengaruhi dropout siswa, termasuk gender, status beasiswa, performa akademik, dan faktor sosial-ekonomi.

2. **Model Prediksi Akurat:** Mengembangkan model Random Forest dengan akurasi 76% yang dapat mengidentifikasi siswa berisiko dropout dengan precision 84%.

3. **Dashboard Interaktif:** Menciptakan dashboard yang user-friendly untuk monitoring real-time performa siswa dan KPI institusi.

4. **Actionable Insights:** Memberikan rekomendasi konkret yang dapat diimplementasikan untuk mengurangi tingkat dropout.

### 9.2 Key Findings

**Faktor Risiko Utama Dropout:**
1. **Performa Akademik Semester Pertama:** Faktor prediktif terkuat
2. **Status Sosial-Ekonomi:** Siswa tanpa beasiswa memiliki risiko dropout lebih tinggi
3. **Gender:** Siswa laki-laki menunjukkan tingkat dropout sedikit lebih tinggi
4. **Usia Pendaftaran:** Siswa yang mendaftar di usia lebih tua memiliki pola completion yang berbeda

**Kekuatan Model:**
- Excellent performance dalam mendeteksi siswa yang akan lulus (92% recall)
- Sangat baik dalam mengidentifikasi siswa dropout (84% precision)
- Dapat digunakan sebagai early warning system yang efektif

### 9.3 Limitasi Penelitian

1. **Temporal Aspect:** Model saat ini tidak mempertimbangkan aspek temporal secara eksplisit
2. **External Factors:** Faktor eksternal seperti kondisi ekonomi keluarga atau kesehatan tidak tercakup
3. **Class Imbalance:** Kelas "Enrolled" memiliki performa prediksi yang lebih rendah
4. **Feature Engineering:** Masih ada ruang untuk pengembangan fitur yang lebih sophisticated

### 9.4 Recommendations Action Items

#### 9.4.1 Action Items Immediate (0-3 bulan)

**1. Implementasi Early Warning System**
- Deploy model prediksi untuk screening siswa baru
- Buat alert system untuk siswa dengan probabilitas dropout >70%
- Training staff akademik untuk menggunakan sistem prediksi

**2. Program Intervensi Targeted**
- Fokus pada siswa laki-laki dengan performa semester pertama rendah
- Prioritaskan siswa tanpa beasiswa untuk program bantuan finansial
- Buat program mentoring khusus untuk siswa berisiko tinggi

**3. Dashboard Implementation**
- Deploy dashboard untuk manajemen dan staff akademik
- Training penggunaan dashboard untuk decision making
- Establish regular review meetings berdasarkan dashboard insights

#### 9.4.2 Action Items Short-term (3-6 bulan)

**1. Program Beasiswa Expansion**
- Tingkatkan alokasi beasiswa berdasarkan analisis ROI
- Buat kriteria beasiswa yang mempertimbangkan risiko dropout
- Develop partnership dengan industri untuk funding tambahan

**2. Academic Support Enhancement**
- Strengthen program remedial untuk semester pertama
- Implement peer tutoring system
- Develop early intervention protocols untuk siswa dengan grade rendah

**3. Data Collection Improvement**
- Tambah variabel sosial-ekonomi yang lebih detail
- Implement regular student survey untuk faktor non-akademik
- Integrate dengan sistem informasi yang lebih comprehensive

#### 9.4.3 Action Items Long-term (6-12 bulan)

**1. Predictive Analytics Enhancement**
- Develop ensemble model dengan multiple algorithms
- Implement real-time scoring system
- Add temporal modeling untuk prediction timeline

**2. Institutional Policy Changes**
- Review admission criteria berdasarkan findings
- Adjust curriculum berdasarkan dropout patterns
- Develop retention-focused KPIs untuk staff evaluation

**3. Technology Infrastructure**
- Implement automated data pipeline
- Develop mobile app untuk student engagement
- Create integrated student success platform

#### 9.4.4 Success Metrics

**Key Performance Indicators untuk monitoring success:**
1. **Dropout Rate Reduction:** Target pengurangan dari 32.1% menjadi <25% dalam 2 tahun
2. **Early Detection Accuracy:** Maintain model accuracy >75%
3. **Intervention Success Rate:** >60% siswa berisiko yang menerima intervensi berhasil lulus
4. **Student Satisfaction:** Peningkatan satisfaction score terkait academic support
5. **Financial Impact:** ROI positif dari program intervensi dalam 18 bulan

#### 9.4.5 Risk Mitigation

**Potential Risks dan Mitigation Strategies:**
1. **Model Drift:** Regular retraining dan performance monitoring
2. **Resource Constraints:** Phased implementation berdasarkan prioritas
3. **Staff Resistance:** Comprehensive training dan change management
4. **Privacy Concerns:** Implement robust data governance dan consent mechanisms
5. **False Positives:** Regular model calibration dan human oversight

---