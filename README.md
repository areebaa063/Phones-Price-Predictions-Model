# 📱 Smartphone Price Prediction (10-Fold Cross Validation)

Predicts the **price (USD) of a smartphone** from its specifications, brand, operating system and release date. Five regression algorithms are compared with **10-fold cross validation** and **RMSE**; the best one is saved as `model.pkl` and served through a **Streamlit** web app.

**Live app:** https://phones-price-prediction-areeba-tahir.streamlit.app/
**GitHub repository:** https://github.com/areebaa063/phones-price-prediction

| | |
|---|---|
| **Developer** | Areeba Tahir (FA24-BED-014) |
| **Supervisor** | Dr. Rao Muhammad Adeel Nawab |
| **GitHub** | https://github.com/areebaa063 |
| **Course assignment** | ML A-3 |

---

## Results

All 5 algorithms were tested on exactly the same 10 folds of the full dataset (1,510 phones after removing 2 duplicates). Lower RMSE is better.

| Rank | Algorithm | Mean RMSE | Std RMSE |
|:---:|---|:---:|:---:|
| 1 | **Extra Trees** | **$225** | $33 |
| 2 | Lasso Regression | $233 | $30 |
| 3 | Ridge Regression | $234 | $30 |
| 4 | XGBoost Regressor | $237 | $36 |
| 5 | AdaBoost Regressor | $316 | $32 |

The average phone price in the data is $338. **Extra Trees** has the lowest mean RMSE and was retrained on all 1,510 phones and saved as `model.pkl`. The top four algorithms are within about $12 of each other, which is smaller than the fold-to-fold standard deviation, so Extra Trees is a small edge rather than a large win. See Step 9.4 of the notebook for the full discussion.

---

## Files in this project

| File | Purpose |
|---|---|
| `phones_price_prediction_10fold.ipynb` | The full notebook: data understanding, pre-processing, 10-fold cross validation of 5 models, final model training, application and feedback phases |
| `model.pkl` | The trained best model (a complete scikit-learn pipeline: imputers, scaler, one-hot encoder and Extra Trees) |
| `app.py` | Streamlit web app that loads `model.pkl` and predicts a phone's price |
| `requirements.txt` | Python libraries with exact versions (Python 3.11) |
| `cleaned_all_phones.csv` | The dataset: 1,512 phones, 20 specification columns + phone name + price |
| `README.md` | This file |

---

## The dataset

`cleaned_all_phones.csv` is a cleaned smartphone-specifications dataset: 1,512 phones from 13 brands (Apple, Google, Honor, Huawei, Lenovo, LG, OnePlus, Oppo, Realme, Samsung, Sony, Vivo, Xiaomi) announced between September 2016 and August 2023, with no missing values.

**Target:** `price(USD)`, which ranges from $40 to $2,300 (median $260).

The notebook turns the raw columns into **22 model inputs**:

| Model input | Source column | Type |
|---|---|---|
| `inches`, `battery`, `ram(GB)`, `weight(g)`, `storage(GB)` | same name | Numeric |
| `res_width`, `res_height` | `resolution` (e.g. `1080x2400`) | Numeric |
| `release_year` | `announcement_date` (year + month/12) | Numeric |
| `os_version` | `os` (e.g. `Android 12` gives 12) | Numeric |
| `video_720p`, `video_1080p`, `video_4K`, `video_8K`, `video_30fps`, `video_60fps`, `video_120fps`, `video_240fps`, `video_480fps`, `video_960fps` | same name (True/False becomes 1/0) | Numeric (0/1) |
| `brand` | same name | Categorical |
| `os_family` | `os` (Android, iOS, EMUI or Tizen) | Categorical |
| `battery_type` | same name | Categorical |

`phone_name` is not used as an input because it is an identifier, not a specification.

---

## How the evaluation works

1. **Pre-processing:** drop `phone_name`, remove duplicate rows, build the 22 features.
2. **Pipeline (inside every fold, so there is no data leakage):** numeric columns are filled with the median and standard-scaled; categorical columns are filled with the most frequent value and one-hot encoded.
3. **10-fold cross validation** (`shuffle=True`, `random_state=42`): the same folds are used for all 5 algorithms. Each fold has 151 test phones and 1,359 training phones.
4. **Metric:** Root Mean Squared Error (RMSE) in US dollars.
5. The algorithm with the lowest mean RMSE is retrained on all the data and saved with `joblib` (compressed, about 1.7 MB).

The 5 algorithms: **Ridge**, **Lasso**, **Extra Trees**, **AdaBoost** and **XGBoost**.

---

## Run the notebook on Google Colab

1. Open https://colab.research.google.com and sign in.
2. **File → Upload notebook** and choose `phones_price_prediction_10fold.ipynb`.
3. **Runtime → Run all**.
4. When Step 2.1 runs, click **Choose Files** and select `cleaned_all_phones.csv`.
5. The notebook finishes in well under a minute and creates `model.pkl` in the Colab **Files** panel. Set `DOWNLOAD_MODEL = True` in Step 10.3 to download it.

---

## Run the app on your own computer

Python 3.11 is recommended (the versions in `requirements.txt` were tested with it).

```bash
# 1. Get the project
git clone https://github.com/areebaa063/phones-price-prediction.git
cd phones-price-prediction

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install the libraries
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

The app opens at http://localhost:8501. Enter the phone's brand, operating system, release date, screen, memory, battery and video features, then click **Predict price**.

> `model.pkl` was trained with the library versions in `requirements.txt` (scikit-learn 1.7.2). Install those exact versions, or re-run the notebook to rebuild `model.pkl` for your versions.

---

## Deploy the app on Streamlit Community Cloud

1. Create a public GitHub repository named **`phones-price-prediction`** under https://github.com/areebaa063 and upload these 6 files to its root: `app.py`, `model.pkl`, `cleaned_all_phones.csv`, `requirements.txt`, `phones_price_prediction_10fold.ipynb` and `README.md`. `model.pkl` must be committed (it is only about 1.7 MB, far below GitHub's file-size limit).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub**, then choose the repository `areebaa063/phones-price-prediction`, branch `main`, and main file `app.py`.
4. Open **Advanced settings**, select **Python 3.11**, and set the custom subdomain to `phones-price-prediction-areeba-tahir`.
5. Click **Deploy**. After a few minutes the app is live at https://phones-price-prediction-areeba-tahir.streamlit.app/.

---

## Notes and limitations

- The model was trained on phones announced between **September 2016 and August 2023**; predictions for later dates are less reliable (the app shows a warning).
- The best model's error is about 67% of the average phone price, because specifications alone do not capture brand value and market positioning. RMSE is also dominated by the few very expensive phones; typical budget and mid-range phones have smaller dollar errors.
- Ideas for improvement (log-price target, hyperparameter tuning, more features such as chipset and camera) are listed in Step 12 of the notebook.

---

**Author:** Areeba Tahir (FA24-BED-014) · https://github.com/areebaa063
