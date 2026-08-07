# FinalProject-KMeans-PSO

## Optimasi Inisialisasi Centroid K-Means Menggunakan Particle Swarm Optimization (PSO)

## Deskripsi Project
Project ini merupakan implementasi optimasi algoritma K-Means menggunakan metode Particle Swarm Optimization (PSO). Metode PSO digunakan untuk menentukan inisialisasi centroid awal sehingga proses clustering pada algoritma K-Means dapat menghasilkan pengelompokan data yang lebih optimal.
Project ini diterapkan pada proses segmentasi pelanggan menggunakan dataset Mall Customers.

## Dataset
Dataset yang digunakan:
Mall Customers Dataset
Atribut yang digunakan:
- Customer ID
- Gender
- Age
- Annual Income
- Spending Score

## Metode yang Digunakan
Tahapan metode pada project ini:
1. Membaca dataset pelanggan.
2. Melakukan preprocessing data.
3. Melakukan normalisasi data menggunakan StandardScaler.
4. Menentukan centroid awal menggunakan Particle Swarm Optimization (PSO).
5. Melakukan proses clustering menggunakan K-Means.
6. Mengevaluasi hasil clustering menggunakan Silhouette Score.
7. Menampilkan hasil visualisasi clustering.

## Teknologi
Project ini dibuat menggunakan:
- Python
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Struktur File
FinalProject-KMeans-PSO
│
├── main.py
├── Mall_Customers (1).csv
├── requirements.txt
└── README.md

## Cara Menjalankan Program
Install library yang dibutuhkan:
pip install -r requirements.txt

## Output Program
Program menghasilkan:
- Hasil clustering pelanggan.
- Visualisasi kelompok pelanggan.
- Nilai evaluasi Silhouette Score.
- Perbandingan hasil clustering menggunakan optimasi PSO.

## Author
Egi-er3j


