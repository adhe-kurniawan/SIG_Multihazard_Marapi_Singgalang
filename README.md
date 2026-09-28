# 🛰️ Sistem Prediksi dan Deteksi Dini Bencana Multi-Hazard
### Integrasi Citra Drone, Data Meteorologi BMKG, dan Sistem Informasi Geografis (GIS)

Studi kasus: DAS Marapi–Singgalang, Sumatera Barat — deteksi dini 3 jenis bencana
(Banjir, Longsor, Galodo) secara simultan.

## Cara Menjalankan Aplikasi

```bash
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi terbuka otomatis di `http://localhost:8501`.

## Struktur Proyek

```
bencana_drone_app/
├── app.py                          # Aplikasi utama (WebGIS + Dashboard + Cetak PDF)
├── requirements.txt
├── data/
│   ├── generate_dataset.py         # Pembangkit GIS multi-hazard & data BMKG
│   ├── generate_drone_images.py    # Pembangkit citra drone sintetis (5 kelas)
│   ├── gis_grid_multihazard.csv
│   ├── stasiun_bmkg.csv
│   ├── data_meteorologi_bmkg.csv
│   ├── dataset_latih_spatiotemporal.csv
│   ├── kecamatan_meta.csv
│   └── drone_images/               # 300 citra drone sintetis
├── models/
│   ├── train_image_classifier.py
│   ├── train_spatiotemporal_model.py
│   ├── *.joblib                    # Model & scaler terlatih
│   ├── artifacts/                  # Grafik evaluasi model fusi
│   └── artifacts_citra/            # Grafik evaluasi model citra
├── report_assets/                  # Aset visual laporan (termasuk tampilan_sig_multihazard.png)
├── sig_multihazard_standalone.html # SIG interaktif mandiri (buka langsung di browser, tanpa perlu instalasi)
├── outputs_demo/                   # Contoh hasil cetak PDF nyata dari sistem
└── report/
    ├── content.js / build.js
    ├── Laporan_Deteksi_Dini_MultiBencana_Drone_BMKG_GIS.docx
    └── Laporan_Deteksi_Dini_MultiBencana_Drone_BMKG_GIS.pdf
```

## Fitur Utama

1. **Analisis Citra Drone** — klasifikasi 5 kelas indikator visual (vegetasi normal, genangan air, retakan tanah, lahan terbuka vulkanik, permukiman) menggunakan Deep Neural Network (akurasi 98,7%).
2. **Dashboard Meteorologi BMKG** — eksplorasi data curah hujan, suhu, kelembaban dari 5 stasiun.
3. **Peta GIS Multi-Hazard** — WebGIS interaktif (prediksi gabungan, kerentanan per jenis, layer GIS dasar).
4. **Prediksi Spatio-Temporal** — proyeksi status bencana 7 hari ke depan per kecamatan.
5. **Sistem Peringatan Dini** — status & rekomendasi tindakan per kecamatan (AMAN/BANJIR/LONGSOR/GALODO).
6. **Cetak Laporan PDF** — unduh ringkasan hasil analisis langsung dari aplikasi (fitur ReportLab).
7. **Evaluasi Model** — metrik kinerja model fusi (akurasi 99,3%, F1-makro 0,981) dan model citra.
8. **SIG Interaktif Mandiri** — buka `sig_multihazard_standalone.html` langsung di browser (tanpa perlu menjalankan Streamlit) untuk melihat peta multi-hazard dengan 4 mode layer (Dominan, Banjir, Longsor, Galodo).

## Membangun Ulang Dataset & Model (opsional)

```bash
python data/generate_dataset.py
python data/generate_drone_images.py
python models/train_image_classifier.py
python models/train_spatiotemporal_model.py
```

## Catatan Penting

Seluruh **citra drone**, **data meteorologi BMKG**, dan **label kejadian bencana** pada
purwarupa ini bersifat **SINTETIS** — dibangkitkan secara prosedural/heuristik berdasarkan
karakteristik fisik nyata wilayah studi, karena tidak tersedia akses ke citra drone
operasional, umpan API BMKG real-time, maupun basis data kejadian BNPB/BPBD terverifikasi
dalam lingkup pengerjaan ini. Gunakan untuk riset, pembelajaran metodologi, dan demonstrasi
sistem — **bukan** untuk pengambilan keputusan operasional kebencanaan aktual.

Dokumentasi metodologi lengkap (Bab I–V), hasil evaluasi, dan panduan penggunaan detail
tersedia pada `report/Laporan_Deteksi_Dini_MultiBencana_Drone_BMKG_GIS.docx` (dan versi
PDF-nya).
