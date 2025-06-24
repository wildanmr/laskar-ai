# Laporan Analisis Attrition Karyawan PT Jaya Jaya Maju

* **Nama:** Wildan Mufid Ramadhan
* **Email:** wildan.20nov@gmail.com
* **Dicoding ID:** wildan.20nov@gmail.com

---

## 1. Domain Proyek

Proyek ini berfokus pada domain Human Resources (HR) Analytics, khususnya dalam mengatasi permasalahan *attrition rate* karyawan di PT Jaya Jaya Maju. Attrition rate, atau tingkat keluar masuk karyawan, merupakan metrik krusial yang mengukur persentase karyawan yang meninggalkan perusahaan dalam periode waktu tertentu. Tingkat attrition yang tinggi dapat berdampak negatif signifikan terhadap operasional perusahaan, termasuk peningkatan biaya rekrutmen dan pelatihan, penurunan produktivitas, hilangnya pengetahuan institusional, dan dampak negatif pada moral karyawan yang tersisa. Oleh karena itu, memahami faktor-faktor pendorong attrition dan mengembangkan strategi retensi yang efektif menjadi prioritas utama bagi departemen HR.

Dalam konteks PT Jaya Jaya Maju, sebuah perusahaan multinasional yang telah berdiri sejak tahun 2000 dengan lebih dari 1000 karyawan, tingkat attrition yang mencapai lebih dari 10% menjadi perhatian serius. Angka ini jauh melampaui rata-rata industri yang sehat, yang umumnya berkisar antara 5-8%. Tujuan utama proyek ini adalah untuk mengidentifikasi akar penyebab di balik tingginya attrition rate ini dan menyediakan solusi berbasis data untuk membantu departemen HR dalam memonitor serta mengurangi angka tersebut. Proyek ini akan melibatkan analisis data ekstensif, visualisasi informasi melalui *business dashboard*, dan pengembangan model *machine learning* sederhana untuk prediksi attrition, yang semuanya bertujuan untuk mendukung pengambilan keputusan yang lebih baik dan proaktif dalam manajemen sumber daya manusia.

---

## 2. Business Understanding

PT Jaya Jaya Maju menghadapi tantangan signifikan terkait retensi karyawan. Meskipun telah menjadi entitas bisnis yang mapan dan besar, perusahaan ini mengalami kesulitan dalam mengelola tenaga kerjanya secara efektif, yang tercermin dari *attrition rate* yang mengkhawatirkan. Manajer departemen HR telah mengidentifikasi masalah ini sebagai prioritas utama dan mencari bantuan untuk memahami mengapa karyawan meninggalkan perusahaan dan bagaimana mereka dapat memitigasi tren ini.

**Permasalahan Bisnis:**

*   **Tingginya Attrition Rate:** Tingkat keluar masuk karyawan di PT Jaya Jaya Maju melebihi 10%, yang secara substansial lebih tinggi dari standar industri. Ini menunjukkan adanya masalah mendasar dalam kepuasan karyawan, lingkungan kerja, atau manajemen talenta.
*   **Dampak Negatif:** Attrition yang tinggi menyebabkan berbagai konsekuensi negatif, seperti:
    *   **Peningkatan Biaya:** Biaya yang terkait dengan rekrutmen, *onboarding*, dan pelatihan karyawan baru sangat tinggi. Selain itu, ada biaya tidak langsung seperti hilangnya produktivitas selama masa transisi.
    *   **Penurunan Produktivitas:** Kepergian karyawan, terutama yang berpengalaman, dapat mengganggu alur kerja, menunda proyek, dan mengurangi efisiensi tim.
    *   **Hilangnya Pengetahuan Institusional:** Karyawan yang pergi membawa serta pengalaman, keahlian, dan pengetahuan unik yang sulit digantikan, berdampak pada inovasi dan daya saing perusahaan.
    *   **Dampak pada Moral Karyawan:** Tingkat turnover yang tinggi dapat menurunkan moral karyawan yang tersisa, menciptakan ketidakpastian, dan bahkan mendorong lebih banyak karyawan untuk mencari peluang di tempat lain.

**Cakupan Proyek:**

Proyek ini bertujuan untuk mengidentifikasi faktor-faktor yang mempengaruhi tingginya attrition rate di perusahaan Jaya Jaya Maju menggunakan pendekatan data science dan machine learning. Dengan pemahaman yang lebih baik tentang faktor-faktor penyebab attrition, perusahaan dapat mengimplementasikan strategi retensi yang lebih efektif dan mengurangi biaya terkait pergantian karyawan.

**Tujuan Bisnis:**

Manajer HR memiliki dua tujuan utama dari proyek ini:

1.  **Identifikasi Faktor Pendorong Attrition:** Mengidentifikasi variabel-variabel kunci dalam data karyawan yang berkorelasi kuat dengan keputusan karyawan untuk meninggalkan perusahaan. Ini akan memberikan pemahaman mendalam tentang akar penyebab masalah.
2.  **Pengembangan Business Dashboard:** Membuat *business dashboard* yang intuitif dan interaktif untuk membantu departemen HR memonitor faktor-faktor yang mempengaruhi *attrition rate* secara berkelanjutan. Dashboard ini harus memungkinkan pemangku kepentingan untuk dengan cepat mengidentifikasi tren, area masalah, dan efektivitas intervensi.

**Manfaat yang Diharapkan:**

Dengan berhasilnya proyek ini, PT Jaya Jaya Maju diharapkan dapat:

*   **Menurunkan Attrition Rate:** Mengurangi persentase karyawan yang keluar ke tingkat yang lebih sehat dan berkelanjutan.
*   **Meningkatkan Retensi Karyawan:** Mengembangkan strategi yang lebih efektif untuk mempertahankan talenta kunci.
*   **Optimasi Biaya HR:** Mengurangi biaya yang terkait dengan turnover karyawan.
*   **Peningkatan Produktivitas dan Moral:** Menciptakan lingkungan kerja yang lebih stabil dan positif, yang pada gilirannya akan meningkatkan produktivitas dan kepuasan karyawan.
*   **Pengambilan Keputusan Berbasis Data:** Memberdayakan departemen HR dengan alat dan wawasan yang diperlukan untuk membuat keputusan yang lebih tepat dan proaktif.

---

## 3. Persiapan

**Sumber Data:** https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee

### 📦Buat Virtual Environment & Install Library

Pastikan Python sudah terinstal di sistem Anda. Jalankan perintah berikut untuk mempersiapkan environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirement.txt
```

### 🚀 Menjalankan Aplikasi Prediksi

Untuk menjalankan aplikasi prediksi menggunakan Streamlit:

```bash
streamlit run prediction_app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`.

### 📊 Menjalankan Metabase

Dashboard Metabase sudah disiapkan dalam direktori `metabase` dan dapat dijalankan menggunakan Docker Compose:

```bash
cd metabase
docker-compose up -d
```

Tunggu beberapa saat hingga proses inisialisasi Metabase selesai. Setelah itu, buka browser dan akses:

```
http://localhost:3000
```

 🔐 Kredensial Login

* **Email**: `root@mail.com`
* **Password**: `root123`

### 📁 Struktur Direktori

```
.
├── metabase/               # Docker Compose untuk Metabase
├── prediction_app.py       # Aplikasi prediksi Streamlit
├── requirement.txt         # Daftar dependensi Python
└── README.md               # Dokumentasi proyek
```

---

## 4. Data Understanding

Data yang digunakan dalam proyek ini disediakan dalam format CSV dengan nama `employee_data.csv`. Dataset ini berisi informasi detail mengenai karyawan PT Jaya Jaya Maju, yang mencakup berbagai atribut demografi, pekerjaan, dan kepuasan kerja. Pemahaman yang mendalam tentang struktur dan konten data ini sangat penting untuk analisis yang akurat dan pengembangan model yang efektif.

**Sumber Data:**

https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee

**Struktur Data:**

Dataset ini terdiri dari 1470 baris (mewakili 1470 karyawan) dan 35 kolom. Setiap baris merepresentasikan satu karyawan, dan setiap kolom adalah atribut spesifik dari karyawan tersebut. Berikut adalah beberapa kolom kunci yang diidentifikasi:

| Kolom                 | Tipe Data   | Deskripsi                                                                 |
| :-------------------- | :---------- | :------------------------------------------------------------------------ |
| `EmployeeId`          | Numerik     | ID unik untuk setiap karyawan.                                            |
| `Age`                 | Numerik     | Usia karyawan.                                                            |
| `Attrition`           | Kategorikal | Menunjukkan apakah karyawan telah meninggalkan perusahaan (Yes/No). Ini adalah variabel target. |
| `BusinessTravel`      | Kategorikal | Frekuensi perjalanan bisnis karyawan (e.g., Travel_Rarely, Travel_Frequently, Non-Travel). |
| `DailyRate`           | Numerik     | Tingkat gaji harian.                                                      |
| `Department`          | Kategorikal | Departemen tempat karyawan bekerja (e.g., Sales, R&D, Human Resources).   |
| `DistanceFromHome`    | Numerik     | Jarak dari rumah ke tempat kerja.                                         |
| `Education`           | Numerik     | Tingkat pendidikan (1-5, e.g., 1=Below College, 5=Master).                 |
| `EducationField`      | Kategorikal | Bidang pendidikan (e.g., Life Sciences, Medical, Marketing).              |
| `EmployeeCount`       | Numerik     | Jumlah karyawan (nilai konstan 1 untuk setiap baris).                     |
| `EnvironmentSatisfaction` | Numerik   | Tingkat kepuasan lingkungan kerja (1-4, e.g., 1=Low, 4=Very High).        |
| `Gender`              | Kategorikal | Jenis kelamin karyawan (Male/Female).                                     |
| `HourlyRate`          | Numerik     | Tingkat gaji per jam.                                                     |
| `JobInvolvement`      | Numerik     | Tingkat keterlibatan dalam pekerjaan (1-4).                               |
| `JobLevel`            | Numerik     | Tingkat pekerjaan (1-5).                                                  |
| `JobRole`             | Kategorikal | Peran pekerjaan karyawan (e.g., Sales Executive, Research Scientist).     |
| `JobSatisfaction`     | Numerik     | Tingkat kepuasan kerja (1-4).                                             |
| `MaritalStatus`       | Kategorikal | Status pernikahan (Single, Married, Divorced).                           |
| `MonthlyIncome`       | Numerik     | Pendapatan bulanan.                                                       |
| `MonthlyRate`         | Numerik     | Tingkat gaji bulanan.                                                     |
| `NumCompaniesWorked`  | Numerik     | Jumlah perusahaan yang pernah dikerjakan sebelumnya.                      |
| `Over18`              | Kategorikal | Apakah karyawan berusia di atas 18 tahun (nilai konstan \'Y\').             |
| `OverTime`            | Kategorikal | Apakah karyawan sering lembur (Yes/No).                                   |
| `PercentSalaryHike`   | Numerik     | Persentase kenaikan gaji terakhir.                                        |
| `PerformanceRating`   | Numerik     | Penilaian kinerja (1-4).                                                  |
| `RelationshipSatisfaction` | Numerik | Tingkat kepuasan hubungan di tempat kerja (1-4).                          |
| `StandardHours`       | Numerik     | Jam kerja standar (nilai konstan 80).                                     |
| `StockOptionLevel`    | Numerik     | Tingkat opsi saham.                                                       |
| `TotalWorkingYears`   | Numerik     | Total tahun pengalaman kerja.                                             |
| `TrainingTimesLastYear` | Numerik   | Jumlah pelatihan yang diikuti tahun lalu.                                 |
| `WorkLifeBalance`     | Numerik     | Tingkat keseimbangan kehidupan kerja (1-4).                               |
| `YearsAtCompany`      | Numerik     | Total tahun bekerja di perusahaan saat ini.                               |
| `YearsInCurrentRole`  | Numerik     | Total tahun dalam peran saat ini.                                         |
| `YearsSinceLastPromotion` | Numerik | Tahun sejak promosi terakhir.                                             |
| `YearsWithCurrManager` | Numerik    | Tahun dengan manajer saat ini.                                            |

**Kualitas Data Awal:**

Berdasarkan pemeriksaan awal, beberapa observasi mengenai kualitas data dapat dicatat:

*   **Missing Values:** Kolom `Attrition` memiliki beberapa nilai yang hilang (NaN), yang perlu ditangani. Penanganan nilai hilang ini akan menjadi bagian penting dari tahap *Data Preparation*.
*   **Kolom Konstan:** Kolom `EmployeeCount`, `StandardHours`, dan `Over18` memiliki nilai yang konstan di seluruh dataset. Kolom-kolom ini tidak akan memberikan informasi yang berguna untuk analisis atau pemodelan, sehingga dapat dihapus.
*   **Tipe Data:** Sebagian besar kolom memiliki tipe data yang sesuai. Kolom kategorikal akan memerlukan *encoding* (misalnya, *Label Encoding* atau *One-Hot Encoding*) untuk dapat digunakan dalam model *machine learning*.
*   **Variabel Target:** Kolom `Attrition` adalah variabel target biner (`Yes`/`No`), yang akan dikonversi menjadi format numerik (misalnya, 1 untuk `Yes` dan 0 untuk `No`) untuk pemodelan.

**Implikasi untuk Analisis:**

Dataset ini kaya akan informasi yang relevan untuk memahami *attrition*. Berbagai atribut seperti demografi (`Age`, `Gender`, `MaritalStatus`), aspek pekerjaan (`Department`, `JobRole`, `JobLevel`, `MonthlyIncome`, `OverTime`), dan faktor kepuasan (`EnvironmentSatisfaction`, `JobSatisfaction`, `RelationshipSatisfaction`, `WorkLifeBalance`) dapat digunakan untuk mengidentifikasi pola dan korelasi dengan *attrition rate*. Analisis eksplorasi akan fokus pada hubungan antara variabel-variabel ini dengan variabel target `Attrition` untuk mengungkap *insights* kunci.

---

## 5. Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) dilakukan untuk memahami karakteristik data, mengidentifikasi pola, dan menemukan hubungan antara variabel-variabel dengan *attrition*. Analisis ini berfokus pada visualisasi dan statistik deskriptif untuk mendapatkan *insights* awal.

### 5.1. Attrition Rate Keseluruhan

Dari 1470 karyawan, 237 di antaranya mengalami *attrition*, menghasilkan *attrition rate* sebesar 16.92%. Angka ini mengkonfirmasi bahwa PT Jaya Jaya Maju menghadapi tantangan signifikan dalam retensi karyawan.

### 5.2. Attrition Berdasarkan Departemen

Visualisasi menunjukkan bahwa departemen Research & Development memiliki *attrition rate* tertinggi, diikuti oleh Sales, dan Human Resources. Ini mengindikasikan bahwa faktor-faktor spesifik di departemen Sales mungkin berkontribusi pada tingginya turnover.

### 5.3. Attrition Berdasarkan Peran Pekerjaan (Job Role)

Analisis *job role* mengungkapkan bahwa Laboratory Technician memiliki *attrition rate* yang sangat tinggi dibandingkan peran lainnya. Ini mungkin disebabkan oleh tekanan kerja yang tinggi, target yang agresif, atau kurangnya dukungan. Peran seperti Sales Executive dan Research Scientist juga menunjukkan tingkat *attrition* yang signifikan.

### 5.4. Attrition Berdasarkan Status Pernikahan (Marital Status)

Karyawan dengan status pernikahan Single menunjukkan *attrition rate* tertinggi. Hal ini bisa jadi karena karyawan single memiliki fleksibilitas lebih besar untuk berpindah pekerjaan atau mencari peluang baru tanpa terikat tanggung jawab keluarga yang sama dengan karyawan menikah.

### 5.5. Attrition Berdasarkan Overtime

Salah satu temuan paling mencolok adalah dampak *overtime*. Karyawan yang sering lembur memiliki *attrition rate* yang jauh lebih tinggi dibandingkan dengan yang tidak lembur. Ini sangat menekankan pentingnya *work-life balance* dan potensi *burnout* sebagai pemicu *attrition*.

### 5.6. Attrition Berdasarkan Gender

Terdapat perbedaan dalam *attrition rate* antara karyawan pria dan wanita, di mana karyawan pria menunjukkan tingkat *attrition* yang lebih tinggi dibandingkan wanita.

### 5.7. Attrition Berdasarkan Lama Bekerja di Perusahaan (YearsAtCompany)

*Attrition rate* cenderung lebih tinggi pada karyawan dengan masa kerja yang relatif singkat (0-2 tahun). Ini menyoroti pentingnya program *onboarding* yang efektif dan strategi retensi awal untuk karyawan baru. Setelah beberapa tahun, *attrition rate* cenderung menurun, menunjukkan stabilitas yang lebih besar pada karyawan yang telah lama bekerja.

### 5.8. Attrition Berdasarkan Pendapatan Bulanan (MonthlyIncome)

Analisis pendapatan bulanan menunjukkan bahwa *attrition rate* cenderung lebih tinggi pada karyawan dengan pendapatan yang lebih rendah. Ini mengindikasikan bahwa kompensasi mungkin menjadi faktor pendorong *attrition* bagi sebagian karyawan.

### 5.9. Attrition Berdasarkan Usia (Age)

Karyawan yang lebih muda (terutama di awal karir) cenderung memiliki *attrition rate* yang lebih tinggi. Ini sejalan dengan temuan *YearsAtCompany*, di mana karyawan baru dan muda mungkin masih mencari jalur karir yang paling sesuai.

### 5.10. Attrition Berdasarkan Tingkat Pekerjaan (JobLevel)

Karyawan di *Job Level* 1 (entry level) memiliki *attrition rate* tertinggi. Ini menunjukkan bahwa karyawan di posisi awal mungkin merasa kurang memiliki prospek karir atau pengembangan, sehingga mencari peluang di tempat lain.

### 5.11. Attrition Berdasarkan Kepuasan Lingkungan Kerja (EnvironmentSatisfaction)

Ada korelasi yang jelas antara kepuasan lingkungan kerja dan *attrition*. Karyawan dengan tingkat kepuasan lingkungan kerja yang rendah memiliki *attrition rate* yang jauh lebih tinggi. Ini menekankan pentingnya menciptakan lingkungan kerja yang positif dan mendukung.

### 5.12. Attrition Berdasarkan Kepuasan Kerja (JobSatisfaction)

Mirip dengan kepuasan lingkungan, kepuasan kerja juga sangat mempengaruhi *attrition*. Karyawan yang tidak puas dengan pekerjaan mereka cenderung lebih mungkin untuk keluar.

### 5.13. Attrition Berdasarkan Keseimbangan Kehidupan Kerja (WorkLifeBalance)

Karyawan dengan *work-life balance* yang buruk memiliki *attrition rate* yang lebih tinggi. Ini memperkuat temuan dari analisis *overtime* bahwa keseimbangan antara pekerjaan dan kehidupan pribadi adalah faktor kunci dalam retensi karyawan.

**Kesimpulan EDA:**

EDA telah mengidentifikasi beberapa faktor kunci yang berkorelasi dengan *attrition*, termasuk departemen, peran pekerjaan, status pernikahan (Single), *overtime*, masa kerja singkat, pendapatan rendah, usia muda, *job level* rendah, dan tingkat kepuasan yang rendah (lingkungan kerja, pekerjaan, *work-life balance*).

## 6. Data Preparation

Tahap *Data Preparation* melibatkan serangkaian proses untuk membersihkan, mengubah, dan mempersiapkan data mentah agar siap untuk analisis lebih lanjut dan pemodelan *machine learning*. Langkah-langkah ini sangat penting untuk memastikan kualitas dan konsistensi data, yang pada gilirannya akan mempengaruhi akurasi model dan validitas *insights* yang dihasilkan.

### 6.1. Penanganan Missing Values

*   **Kolom `Attrition`:** <br>
Kolom ini merupakan variabel target. Beberapa baris memiliki nilai yang hilang (NaN). Untuk tujuan analisis dan pemodelan, nilai-nilai NaN ini diisi dengan `0` (merepresentasikan `No Attrition`). Setelah itu, kolom ini dikonversi menjadi tipe data boolean (`True` untuk `Attrition` dan `False` untuk `No Attrition`). Konversi ini penting karena model *machine learning* memerlukan input numerik atau boolean.

### 6.2. Penghapusan Kolom Tidak Relevan

Beberapa kolom dalam dataset diidentifikasi tidak memberikan informasi yang berarti untuk analisis atau pemodelan karena memiliki nilai yang konstan di seluruh dataset atau merupakan pengidentifikasi unik yang tidak relevan sebagai fitur prediktif. Kolom-kolom ini dihapus untuk mengurangi dimensi data, mempercepat proses komputasi, dan menghindari *noise* yang tidak perlu. Kolom yang dihapus meliputi:

*   **`EmployeeCount`**: Selalu bernilai 1.
*   **`StandardHours`**: Selalu bernilai 80.
*   **`Over18`**: Selalu bernilai \'Y\'.
*   **`EmployeeId`**: Merupakan ID unik untuk setiap karyawan, tidak memiliki nilai prediktif untuk *attrition*.

### 6.3. Encoding Variabel Kategorikal

Model *machine learning* umumnya memerlukan input numerik. Oleh karena itu, variabel-variabel kategorikal (tipe data `object`) dalam dataset perlu diubah menjadi representasi numerik. Dalam proyek ini, *Label Encoding* digunakan untuk mengonversi kolom-kolom kategorikal. Setiap kategori unik dalam sebuah kolom akan diberi nilai integer yang unik. Kolom-kolom yang di-encode meliputi:

*   `BusinessTravel`
*   `Department`
*   `EducationField`
*   `Gender`
*   `JobRole`
*   `MaritalStatus`
*   `OverTime`

>**Catatan:** <br>
Meskipun *One-Hot Encoding* seringkali lebih disukai untuk variabel kategorikal nominal karena menghindari asumsi urutan, *Label Encoding* dipilih di sini untuk menjaga kesederhanaan model dan mengurangi dimensi fitur, mengingat instruksi untuk menggunakan model yang sederhana. Untuk model yang lebih kompleks atau jika asumsi urutan menjadi masalah, *One-Hot Encoding* akan menjadi pilihan yang lebih baik.

### 6.4. Pembagian Data (Fitur dan Target)

Setelah pra-pemrosesan, dataset dibagi menjadi fitur (variabel independen, `X`) dan variabel target (variabel dependen, `y`). Variabel target adalah kolom `Attrition` yang telah dikonversi menjadi boolean. Fitur `X` mencakup semua kolom lain yang tersisa setelah penghapusan kolom tidak relevan.

Data kemudian dibagi lagi menjadi *training set* dan *testing set* menggunakan fungsi `train_test_split` dari `sklearn.model_selection`. Proporsi pembagian yang digunakan adalah 80% untuk *training* dan 20% untuk *testing*. Parameter `random_state` diatur untuk memastikan reproduktifitas hasil, dan `stratify=y` digunakan untuk memastikan bahwa proporsi kelas target (`Attrition` vs `No Attrition`) tetap sama di *training* dan *testing set*. Ini sangat penting karena dataset *attrition* seringkali tidak seimbang (jumlah karyawan yang tidak *attrition* jauh lebih banyak daripada yang *attrition*).

### 6.5. Scaling Fitur Numerik

Fitur-fitur numerik dalam dataset memiliki skala yang berbeda-beda (misalnya, `MonthlyIncome` memiliki rentang nilai yang jauh lebih besar daripada `DistanceFromHome`). Model *machine learning* tertentu, seperti *Logistic Regression* yang sensitif terhadap skala fitur, dapat berkinerja lebih baik jika fitur-fitur numerik dinormalisasi atau distandarisasi. Dalam proyek ini, `StandardScaler` dari `sklearn.preprocessing` digunakan untuk menstandarisasi fitur numerik. `StandardScaler` mengubah data sehingga memiliki rata-rata 0 dan standar deviasi 1. Proses *scaling* ini diterapkan hanya pada *training set* (`fit_transform`) dan kemudian *transformasi* yang sama diterapkan pada *testing set* (`transform`) untuk menghindari *data leakage*.

Kolom numerik yang di-scale meliputi:

*   `Age`
*   `DailyRate`
*   `DistanceFromHome`
*   `Education`
*   `EnvironmentSatisfaction`
*   `HourlyRate`
*   `JobInvolvement`
*   `JobLevel`
*   `JobSatisfaction`
*   `MonthlyIncome`
*   `MonthlyRate`
*   `NumCompaniesWorked`
*   `PercentSalaryHike`
*   `PerformanceRating`
*   `RelationshipSatisfaction`
*   `StockOptionLevel`
*   `TotalWorkingYears`
*   `TrainingTimesLastYear`
*   `WorkLifeBalance`
*   `YearsAtCompany`
*   `YearsInCurrentRole`
*   `YearsSinceLastPromotion`
*   `YearsWithCurrManager`

Dengan selesainya tahap *Data Preparation*, data kini berada dalam format yang bersih, konsisten, dan siap untuk digunakan dalam tahap pemodelan *machine learning*.

## 7. Modeling

Tahap *Modeling* melibatkan pemilihan, pelatihan, dan penyetelan model *machine learning* untuk memprediksi *attrition* karyawan. Berdasarkan instruksi untuk menggunakan model yang sederhana, *Logistic Regression* dipilih sebagai algoritma pemodelan.

### 7.1. Pemilihan Algoritma: Logistic Regression

*Logistic Regression* adalah algoritma klasifikasi linier yang kuat dan mudah diinterpretasikan, cocok untuk masalah klasifikasi biner seperti prediksi *attrition* (karyawan *attrition* atau tidak *attrition*). Meskipun namanya mengandung kata "regresi", *Logistic Regression* digunakan untuk klasifikasi. Algoritma ini memodelkan probabilitas suatu kelas menggunakan fungsi logistik (sigmoid).

**Alasan Pemilihan Logistic Regression:**

*   **Sederhana dan Efisien:** *Logistic Regression* relatif sederhana dalam konsep dan komputasi, membuatnya cepat untuk dilatih dan diinterpretasikan.
*   **Interpretasi:** Koefisien model dapat diinterpretasikan untuk memahami bagaimana setiap fitur mempengaruhi probabilitas *attrition*, yang sangat berharga untuk *business understanding*.
*   **Baseline Model:** Sering digunakan sebagai *baseline model* karena kinerjanya yang solid dan kemampuannya untuk memberikan wawasan awal tentang hubungan antara fitur dan target.
*   **Persyaratan Tugas:** Sesuai dengan permintaan untuk menggunakan model *machine learning* yang sederhana.

### 7.2. Pelatihan Model

Model *Logistic Regression* dilatih menggunakan *training set* (`X_train` dan `y_train`) yang telah dipersiapkan pada tahap *Data Preparation*. Parameter `solver=\'liblinear\'` dipilih karena cocok untuk dataset kecil dan mendukung regularisasi L1/L2. `random_state=42` diatur untuk memastikan hasil yang dapat direproduksi.

Proses pelatihan melibatkan penyesuaian bobot (koefisien) untuk setiap fitur sehingga model dapat meminimalkan kesalahan dalam memprediksi probabilitas *attrition*.

### 7.3. Penyimpanan Model dan Artefak

Setelah model dilatih, sangat penting untuk menyimpan model dan objek pra-pemrosesan yang terkait. Ini memungkinkan model untuk digunakan kembali di masa mendatang untuk inferensi (membuat prediksi pada data baru) tanpa perlu melatih ulang model atau melakukan pra-pemrosesan dari awal. Artefak yang disimpan meliputi:

*   **`logistic_regression_model.pkl`**: Objek model *Logistic Regression* yang telah dilatih. Ini berisi semua bobot dan bias yang dipelajari oleh model.
*   **`scaler.pkl`**: Objek `StandardScaler` yang digunakan untuk menstandarisasi fitur numerik. Objek ini harus digunakan untuk menstandarisasi data baru sebelum melakukan prediksi, memastikan konsistensi pra-pemrosesan.
*   **`model_features.pkl`**: Daftar nama kolom (fitur) yang digunakan untuk melatih model. Ini penting untuk memastikan bahwa data baru yang akan diprediksi memiliki urutan dan nama fitur yang sama dengan data pelatihan.
*   **`categorical_cols.pkl`**: Daftar nama kolom kategorikal yang di-encode. Ini membantu dalam proses pra-pemrosesan data baru.
*   **`label_encoders_classes.pkl`**: Sebuah dictionary yang berisi kelas-kelas unik yang ditemukan oleh `LabelEncoder` untuk setiap kolom kategorikal. Ini memastikan bahwa *encoding* yang sama diterapkan pada data baru, bahkan jika data baru tidak mengandung semua kategori yang ada di data pelatihan.

Penyimpanan artefak ini dilakukan menggunakan modul `joblib`, yang efisien untuk menyimpan objek Python yang berisi array NumPy besar.

Dengan model yang telah dilatih dan artefak yang disimpan, kita dapat melanjutkan ke tahap evaluasi untuk menilai kinerja model dan kemudian menggunakannya untuk inferensi pada data karyawan baru.

## 8. Evaluation

Tahap evaluasi sangat penting untuk menilai seberapa baik kinerja model *machine learning* yang telah dilatih dalam memprediksi *attrition*. Metrik evaluasi yang tepat akan memberikan gambaran yang jelas tentang kekuatan dan kelemahan model, serta membantu dalam memahami implikasi praktisnya.

### 8.1. Metrik Evaluasi

Untuk masalah klasifikasi biner seperti prediksi *attrition*, beberapa metrik evaluasi kunci digunakan:

*   **Accuracy (Akurasi):** Proporsi prediksi yang benar dari total prediksi. Ini adalah metrik yang mudah dipahami, tetapi bisa menyesatkan pada dataset yang tidak seimbang (seperti dataset *attrition* di mana kelas \'No Attrition\' jauh lebih dominan).
*   **Precision (Presisi):** Dari semua kasus yang diprediksi sebagai positif (misalnya, *attrition*), berapa banyak yang sebenarnya positif. Presisi tinggi berarti model memiliki sedikit *false positives*.
*   **Recall (Sensitivitas/Cakupan):** Dari semua kasus yang sebenarnya positif (misalnya, *attrition*), berapa banyak yang berhasil diprediksi sebagai positif. *Recall* tinggi berarti model memiliki sedikit *false negatives*.
*   **F1-Score:** Rata-rata harmonik dari *Precision* dan *Recall*. Ini adalah metrik yang baik ketika ada kebutuhan untuk menyeimbangkan *Precision* dan *Recall*, terutama pada kelas yang tidak seimbang.
*   **Confusion Matrix (Matriks Konfusi):** Sebuah tabel yang menunjukkan kinerja model klasifikasi pada *testing set* di mana nilai sebenarnya diketahui. Matriks ini menampilkan jumlah *True Positives (TP)*, *True Negatives (TN)*, *False Positives (FP)*, dan *False Negatives (FN)*.
    *   **TP (True Positives):** Karyawan yang diprediksi *attrition* dan memang *attrition*.
    *   **TN (True Negatives):** Karyawan yang diprediksi tidak *attrition* dan memang tidak *attrition*.
    *   **FP (False Positives):** Karyawan yang diprediksi *attrition* tetapi sebenarnya tidak *attrition* (Type I error).
    *   **FN (False Negatives):** Karyawan yang diprediksi tidak *attrition* tetapi sebenarnya *attrition* (Type II error).

### 8.2. Hasil Evaluasi Model Logistic Regression

Setelah melatih model *Logistic Regression* pada *training set* dan melakukan prediksi pada *testing set*, berikut adalah hasil evaluasinya:

```
Accuracy: 0.90

Classification Report:
              precision    recall  f1-score   support

       False       0.90      1.00      0.95       258
        True       0.89      0.22      0.36        36

    accuracy                           0.90       294
   macro avg       0.90      0.61      0.65       294
weighted avg       0.90      0.90      0.87       294

Confusion Matrix:
[[257   1]
 [ 28   8]]
```

**Interpretasi Hasil:**

*   **Accuracy (0.90):** Model mencapai akurasi 90%. Ini terlihat tinggi, tetapi karena dataset tidak seimbang (hanya sekitar 12% karyawan yang *attrition*), akurasi saja tidak cukup untuk menilai kinerja model secara komprehensif.

*   **Classification Report:**
    *   **Kelas `False` (No Attrition):**
        *   Precision: 0.90 (90% dari prediksi \'No Attrition\' adalah benar).
        *   Recall: 1.00 (100% dari karyawan yang sebenarnya tidak *attrition* berhasil diprediksi dengan benar). Ini menunjukkan model sangat baik dalam mengidentifikasi karyawan yang akan bertahan.
        *   F1-Score: 0.95.
    *   **Kelas `True` (Attrition):**
        *   Precision: 0.89 (89% dari prediksi \'Attrition\' adalah benar). Ini berarti ketika model memprediksi seseorang akan *attrition*, kemungkinan besar itu benar.
        *   Recall: 0.22 (Hanya 22% dari karyawan yang sebenarnya *attrition* berhasil diidentifikasi oleh model). Ini adalah kelemahan utama model. Model ini memiliki banyak *false negatives*, artinya banyak karyawan yang sebenarnya akan *attrition* tidak terdeteksi.
        *   F1-Score: 0.36. F1-score yang rendah untuk kelas \'True\' mengkonfirmasi kinerja yang kurang optimal dalam mendeteksi kasus *attrition*.

*   **Confusion Matrix:**
    *   **TP = 8:** Model dengan benar memprediksi 8 karyawan akan *attrition*.
    *   **TN = 257:** Model dengan benar memprediksi 257 karyawan tidak akan *attrition*.
    *   **FP = 1:** Model salah memprediksi 1 karyawan akan *attrition* (padahal tidak).
    *   **FN = 28:** Model salah memprediksi 28 karyawan tidak akan *attrition* (padahal sebenarnya *attrition*).

**Kesimpulan Evaluasi:**

Model *Logistic Regression* ini menunjukkan kinerja yang sangat baik dalam memprediksi karyawan yang akan bertahan (kelas `False`). Namun, model ini kurang efektif dalam mengidentifikasi karyawan yang akan *attrition* (kelas `True`), dengan *recall* yang rendah (0.22) dan jumlah *false negatives* yang tinggi (28). Ini berarti model cenderung melewatkan banyak kasus *attrition* yang sebenarnya. 

Dalam konteks bisnis, *false negatives* (karyawan yang akan *attrition* tetapi tidak terdeteksi) bisa lebih merugikan daripada *false positives* (karyawan yang diprediksi *attrition* tetapi tidak). Departemen HR ingin mengidentifikasi karyawan yang berisiko tinggi untuk *attrition* agar dapat melakukan intervensi. Model saat ini mungkin tidak cukup sensitif untuk tujuan tersebut. 

Untuk perbaikan di masa mendatang, strategi seperti *oversampling* kelas minoritas, *undersampling* kelas mayoritas, penggunaan *cost-sensitive learning*, atau eksplorasi algoritma lain yang lebih canggih (misalnya, *Random Forest*, *Gradient Boosting*) dapat dipertimbangkan untuk meningkatkan *recall* pada kelas *attrition*.

## 9. Business Dashboard

*Business dashboard* yang telah dikembangkan berfungsi sebagai alat visual interaktif bagi departemen HR PT Jaya Jaya Maju untuk memonitor dan memahami faktor-faktor yang berkontribusi terhadap *attrition rate*. Dashboard ini dirancang untuk menyajikan *insights* dari analisis data secara ringkas dan mudah dicerna, memungkinkan pengambilan keputusan yang lebih cepat dan berbasis data.

### 9.1. Tujuan Dashboard

*   **Visualisasi Data Kunci:** Menyajikan metrik dan visualisasi terkait *attrition* secara jelas dan intuitif.
*   **Identifikasi Tren:** Memungkinkan departemen HR untuk dengan cepat mengidentifikasi tren *attrition* berdasarkan berbagai dimensi (departemen, peran, demografi, dll.).
*   **Pemahaman Faktor Pendorong:** Membantu dalam memahami faktor-faktor spesifik yang memiliki korelasi kuat dengan *attrition*.
*   **Dukungan Pengambilan Keputusan:** Menyediakan informasi yang diperlukan untuk merumuskan strategi retensi karyawan yang efektif dan tepat sasaran.

### 9.2. Manfaat bagi Departemen HR

Dashboard ini memberdayakan departemen HR dengan beberapa cara:

*   **Pemantauan Berkelanjutan:** Memungkinkan pemantauan *attrition rate* dan faktor-faktor terkait secara *real-time* (jika data diperbarui secara berkala) atau periodik.
*   **Identifikasi Dini Masalah:** Dengan visualisasi yang jelas, masalah potensial dapat diidentifikasi lebih awal, memungkinkan intervensi proaktif.
*   **Komunikasi Efektif:** Menyediakan alat yang efektif untuk mengkomunikasikan temuan dan rekomendasi kepada manajemen atau pemangku kepentingan lainnya.
*   **Pengambilan Keputusan Berbasis Bukti:** Mendukung keputusan strategis HR dengan data dan *insights* yang kuat, mengurangi ketergantungan pada intuisi semata.

Dashboard ini merupakan langkah penting bagi PT Jaya Jaya Maju dalam mengadopsi pendekatan berbasis data untuk manajemen sumber daya manusia dan secara proaktif mengatasi tantangan *attrition*.

## 10. Conclusion

Analisis komprehensif terhadap data karyawan PT Jaya Jaya Maju telah mengungkap berbagai faktor signifikan yang berkontribusi pada tingginya *attrition rate* perusahaan. Dengan *attrition rate* keseluruhan sebesar 16.1%, yang jauh di atas rata-rata industri, masalah ini memerlukan perhatian dan tindakan strategis segera.

**Ringkasan Temuan Kunci:**

*   **Departemen Research & Development** dan **peran Laboratory Technician** menunjukkan *attrition rate* tertinggi, mengindikasikan adanya tekanan atau kondisi kerja spesifik di area ini yang perlu diinvestigasi lebih lanjut.
*   **Overtime** adalah pemicu *attrition* yang sangat kuat. Karyawan yang sering lembur memiliki kemungkinan *attrition* yang jauh lebih tinggi, menyoroti pentingnya *work-life balance*.
*   **Karyawan Single** dan **karyawan dengan masa kerja 0-2 tahun** memiliki *attrition rate* yang lebih tinggi, menunjukkan perlunya program retensi dan *onboarding* yang lebih kuat untuk segmen ini.
*   **Kepuasan Lingkungan Kerja, Kepuasan Kerja, dan Keseimbangan Kehidupan Kerja** berkorelasi kuat dengan *attrition*. Tingkat kepuasan yang rendah secara konsisten dikaitkan dengan *attrition rate* yang lebih tinggi.
*   **Job Level 1 (entry level)** juga menunjukkan *attrition rate* yang tinggi, menyiratkan kebutuhan akan jalur karir yang jelas dan peluang pengembangan.

**Rekomendasi Action :**

Berdasarkan *insights* ini, delapan rekomendasi strategis telah dirumuskan untuk membantu PT Jaya Jaya Maju mengurangi *attrition rate*:

1.  **Program Work-Life Balance:** Implementasi kebijakan yang membatasi *overtime* dan mempromosikan fleksibilitas kerja.
2.  **Perbaikan Lingkungan Kerja:** Lakukan survei kepuasan berkala dan tindak lanjuti *feedback* karyawan untuk menciptakan lingkungan yang lebih positif.
3.  **Program Onboarding dan Mentoring:** Kembangkan program komprehensif untuk karyawan baru guna meningkatkan adaptasi dan retensi awal.
4.  **Career Development Path:** Buat jalur karir yang jelas dan program pengembangan terstruktur, terutama untuk posisi *entry level*.
5.  **Fokus pada Departemen Research & Development:** Lakukan analisis mendalam dan intervensi spesifik untuk mengatasi masalah *attrition* di departemen Research & Development.
6.  **Program Retensi untuk Karyawan Single:** Kembangkan program khusus yang menarik bagi karyawan single.
7.  **Regular Employee Satisfaction Survey:** Lakukan survei kepuasan karyawan secara berkala untuk identifikasi masalah dini.
8.  **Compensation Review:** Tinjau struktur kompensasi untuk memastikan daya saing, terutama untuk posisi *entry level* dan departemen dengan *attrition* tinggi.

**Keterbatasan Model Machine Learning:**

Model *Logistic Regression* yang dikembangkan, meskipun mencapai akurasi 90%, menunjukkan keterbatasan dalam mendeteksi kasus *attrition* yang sebenarnya (recall rendah untuk kelas positif). Ini berarti model cenderung menghasilkan banyak *false negatives* (karyawan yang akan *attrition* tetapi tidak terdeteksi). Dalam konteks bisnis, ini berarti departemen HR mungkin melewatkan banyak karyawan yang berisiko tinggi untuk *attrition*. Untuk pengembangan di masa depan, diperlukan eksplorasi model yang lebih canggih atau teknik penanganan *imbalanced data* untuk meningkatkan sensitivitas model terhadap kelas *attrition*.

**Langkah Selanjutnya:**

*   **Implementasi Rekomendasi:** Prioritaskan dan implementasikan rekomendasi yang telah diuraikan, dimulai dengan yang memiliki dampak potensial terbesar.
*   **Monitoring Berkelanjutan:** Gunakan *business dashboard* yang telah disediakan untuk memantau efektivitas intervensi dan tren *attrition* secara berkelanjutan.
*   **Iterasi Model:** Lakukan iterasi pada model *machine learning* dengan data yang lebih banyak, fitur yang direkayasa, atau algoritma yang berbeda untuk meningkatkan kemampuan prediksi, terutama dalam mengidentifikasi karyawan berisiko tinggi.
*   **Pengumpulan Data Tambahan:** Pertimbangkan untuk mengumpulkan data kualitatif (misalnya, melalui *exit interview* yang lebih mendalam atau survei kepuasan yang lebih rinci) untuk mendapatkan pemahaman yang lebih kaya tentang alasan di balik *attrition*.

Dengan pendekatan berbasis data ini, PT Jaya Jaya Maju dapat secara proaktif mengelola *attrition rate*, meningkatkan retensi karyawan, dan pada akhirnya memperkuat fondasi sumber daya manusianya untuk pertumbuhan jangka panjang.