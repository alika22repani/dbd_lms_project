# Peramalan Kasus DBD Bulanan di Indonesia dengan LSTM (TA-04)

Data: OpenDengue (https://opendengue.org/data.html). Model: LSTM (TensorFlow/Keras). Web: Flask.

## Struktur
- `TA04_DBD_LSTM.ipynb` : EDA, preprocessing, pelatihan, evaluasi, penyimpanan artefak
- `app.py`, `templates/index.html` : aplikasi Flask
- `artifacts/` : dibuat oleh notebook (model `.keras`, `scaler.joblib`, `monthly_series.csv`, `test_predictions.csv`, `meta.json`)
- `requirements.txt`, `Procfile`, `render.yaml` : deployment

## Langkah
1. Unduh CSV OpenDengue, taruh di `data/` (mis. `data/opendengue.csv`).
2. `pip install -r requirements.txt jupyter matplotlib`
3. Jalankan notebook dari awal sampai akhir (sesuaikan `DATA_PATH`). Folder `artifacts/` akan terisi.
4. Uji lokal: `python app.py` lalu buka http://localhost:5000
5. Push ke GitHub **termasuk folder `artifacts/`**, lalu deploy di Render (New > Blueprint, atau Web Service dengan start command dari `Procfile`).

## Catatan
- Free tier Render bisa lambat saat pertama diakses (cold start + load TensorFlow).
- Jika Python/TensorFlow lokal Anda berbeda versi dari `requirements.txt`, latih ulang model dengan versi yang sama dengan yang dipakai saat deploy.
