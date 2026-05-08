# Eksperimen SML — Muhammad Fadhil Rizki

Repositori untuk **Kriteria 1: Eksperimen Dataset** — Kelas Membangun Sistem Machine Learning (Dicoding).

- **Nama Siswa:** Muhammad Fadhil Rizki
- **Username Dicoding:** `fadhilspooky`
- **Dataset:** California Housing (regresi) — `sklearn.datasets.fetch_california_housing`
- **Target:** `MedHouseValue` (median harga rumah, $100k)

## Struktur

```
Eksperimen_SML_Muhammad-Fadhil-Rizki/
├── .github/workflows/preprocessing.yml      # CI: jalankan preprocessing otomatis
├── housing_raw/housing.csv                  # Data mentah (snapshot dari sklearn)
├── housing_preprocessing/                   # Output siap-latih (di-generate CI)
│   ├── housing_train.csv / housing_test.csv
│   └── X_train.csv / X_test.csv / y_train.csv / y_test.csv
├── preprocessing/
│   ├── Eksperimen_Muhammad-Fadhil-Rizki.ipynb   # Notebook EDA + preprocessing
│   └── automate_Muhammad-Fadhil-Rizki.py        # Script otomatisasi modular
├── requirements.txt
└── README.md
```

## Menjalankan secara lokal

```bash
pip install -r requirements.txt

# Eksekusi notebook
jupyter nbconvert --to notebook --execute \
  preprocessing/Eksperimen_Muhammad-Fadhil-Rizki.ipynb \
  --output preprocessing/Eksperimen_Muhammad-Fadhil-Rizki.ipynb

# Atau jalankan script automasi
python preprocessing/automate_Muhammad-Fadhil-Rizki.py \
  --raw housing_raw/housing.csv \
  --out housing_preprocessing
```

## GitHub Actions

Workflow `preprocessing.yml` ter-trigger pada:
- `push` ke `main` (jika ada perubahan di `preprocessing/`, `housing_raw/`, atau workflow itu sendiri)
- `workflow_dispatch` (manual run dari tab Actions)

Tiap run akan:
1. Setup Python 3.12.7
2. Install dependencies
3. Jalankan `automate_Muhammad-Fadhil-Rizki.py`
4. Upload `housing_preprocessing/` sebagai artifact (retensi 30 hari)
5. Commit hasil ke repo (jika ada perubahan)

## Tahapan Preprocessing

1. Hapus duplikat
2. Imputasi missing values (median)
3. IQR capping pada fitur skewed (`AveRooms`, `AveBedrms`, `AveOccup`, `Population`, `TotalBedrooms_proxy`)
4. Feature engineering: `RoomsPerHousehold`, `BedroomsPerRoom`, `PopulationPerHousehold`, `IncomePerRoom`
5. Binning `MedInc` → `IncomeCategory` untuk stratified split
6. Train-test split 80/20 (stratified)
7. Standardisasi (`StandardScaler`) — fit di train, transform di test
