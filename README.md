# 🛡️ SQL Injection Detection using Chi-Square Feature Selection & Machine Learning

Tái hiện thí nghiệm từ bài báo:

> **"Enhanced SQL injection detection using chi-square feature selection and machine learning classifiers"**  
> Emanuel Casmiry, Neema Mduma, Ramadhani Sinde (2025)

**Kết quả chính:** Decision Tree đạt **99.73% accuracy** sau khi áp dụng Chi-square feature selection (giảm 95% features: 49,607 → 2,551)

---

## 📌 Tính năng chính

- ✅ Chi-square Feature Selection tự động
- ✅ 5 Machine Learning Classifiers (DT, MNB, SVM, LR, KNN)
- ✅ Data Analysis Tools (7 sections phân tích)
- ✅ Stratified 5-Fold Cross Validation
- ✅ Visualization đầy đủ (8 charts)

---

## 🚀 Quick Start (3 bước)

```bash
# 1. Làm sạch dataset
python clean_data.py

# 2. Phân tích dataset
python data_analysis.py

# 3. Chạy thực nghiệm
python main.py
```

**Kết quả:** 8 charts (PNG) + metrics in console + cleaned dataset

---

## 📂 Cấu trúc thư mục

```
.
├── main.py                    # Main experiment
├── clean_data.py              # Data cleaning
├── data_analysis.py           # Data analysis (7 sections)
├── requirements.txt
└── data/
    ├── SQLiV3.csv            # Original (Kaggle)
    └── SQLiV3_cleaned.csv    # Cleaned (auto-generated)
```

---

## 🛠️ Cài đặt

```bash
# Tạo virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt

# Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"
```

---

## 🗃️ Dataset

### Nguồn dữ liệu

**SQLiV3.csv** từ [Kaggle](https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset)

| Sentence       | Label |
|----------------|-------|
| SQL query text | 0/1   |

- `0` = Normal query (Benign)
- `1` = SQL Injection (Malicious)

### ⚠️ Vấn đề và Giải pháp

**Vấn đề:** Dataset gốc có thể chứa:
- 42 labels thay vì 2
- Missing values, duplicates
- Imbalanced data (ratio 19268:1)

**Giải pháp:** Chạy `clean_data.py` trước
- ✅ Filter chỉ giữ label 0 và 1
- ✅ Xóa duplicates, missing values
- ✅ Balance data → ~30,000 samples (50-50)

---

## 📊 Phân tích Dataset

```bash
python data_analysis.py
```

### PHẦN 1: Khảo sát cơ bản Dataset

**Kích thước dataset:**
- Số dòng (samples): 30,405
- Số cột (features): 2

**Các cột trong dataset:**

| # | Column   | Non-Null Count | Dtype  |
|---|----------|----------------|--------|
| 0 | Sentence | 30,405 non-null | object |
| 1 | Label    | 30,405 non-null | int64  |

- Memory usage: ~3 MB
- Missing values: Không có missing values

**5 dòng đầu tiên:**

| | Sentence | Label |
|---|----------|-------|
| 0 | `" or pg_sleep  (  __TIME__  )  --` | 1 |
| 1 | `AND 1 = utl_inaddr.get_host_address ( ...` | 1 |
| 2 | `select * from users where id = '1' or @@1 ...` | 1 |
| 3 | `select * from users where id = 1 or 1#" ( ...` | 1 |
| 4 | `select name from syscolumns where id = ...` | 1 |

### Key Insights

| Category | Metric | Detail |
|----------|--------|--------|
| **Dataset** | Total samples | ~30,000 queries (balanced 50-50) |
| | Avg length (Benign) | 80 chars |
| | Avg length (Malicious) | 150 chars (2x longer) |
| **Attack Types** | Comment-based | 70% — `--`, `#`, `/* */` |
| | Boolean-based | 60% — `OR 1=1`, `AND 1=1` |
| | UNION-based | 40% — `UNION SELECT` |
| | Time-based | 15% — `SLEEP()`, `WAITFOR` |
| | Error-based | 11% — `CAST`, `CONVERT` |
| | Stacked queries | 8% — `;` multiple statements |
| **Top Words** | Benign | `select`, `from`, `where`, `id` |
| | Malicious | `union`, `sleep`, `or`, `and`, `convert` |

---

## 🧠 Experiment Workflow (`main_imporoved.py`)

```bash
python main_imporoved.py
```

### Step 1: Load & Preprocess

- Loaded **30,405** samples (Benign: 19,128 — 62.91%, Malicious: 11,277 — 37.09%)
- Text cleaning: lowercase, giữ SQL-specific tokens (`'`, `--`, `;`, `|`)
- Remove stop words (giữ SQL keywords: `select`, `or`, `and`, `union`...)

### Step 2: Train-Test Split

| | Size |
|---|---|
| Train | 24,324 (80%) |
| Test | 6,081 (20%) |

### Step 3: TF-IDF Vectorization

- Vocabulary size: **20,844**
- Train shape: (24,324 x 20,844)

### Step 4: Chi-Square Feature Selection

- Optimal k = **2,551** (from paper)
- Features reduced: 20,844 → 2,551 (**87.8% reduction**)

### Step 5: Cross-Validation Results (Stratified 5-Fold)

| Model | Before FS | After FS | Change |
|-------|-----------|----------|--------|
| MNB | 94.64% ± 0.18% | 93.56% ± 0.08% | -1.08% |
| LR | 94.10% ± 0.07% | 94.17% ± 0.12% | +0.07% |
| **DT** | 78.91% ± 0.25% | **99.51% ± 0.11%** ⭐ | **+20.60%** |
| SVM | 97.45% ± 0.17% | 97.84% ± 0.18% | +0.39% |
| KNN | 49.47% ± 0.43% | 91.48% ± 0.48% | +42.01% |

### Step 6: Test Set Evaluation (Best Model: Decision Tree)

| Metric | Score |
|--------|-------|
| Accuracy | **98.37%** |
| Precision | 99.40% |
| Recall | 96.19% |
| F1-Score | 97.77% |
| FPR | 0.34% |
| Misclassification | 1.63% |

### Step 7: Computational Efficiency (Decision Tree)

| Metric | Before FS | After FS | Improvement |
|--------|-----------|----------|-------------|
| Training Time | 1.6921s | 0.1213s | **93% faster** |
| Inference Time | 0.0020 ms/query | 0.0001 ms/query | **95% faster** |
| Model Size | 0.79 MB | 0.03 MB | **96% smaller** |

---

## 📈 Visualization

Output: `results/`

| File | Mô tả |
|------|--------|
| `comparison.png` | Grouped bar chart — Before vs After FS |
| `tsne_comparison.png` | t-SNE 2D — Before vs After FS |

![Model Comparison](results/comparison.png)
*Before vs After Feature Selection*

![t-SNE](results/tsne_comparison.png)
*t-SNE: Before FS (overlap) vs After FS (tách biệt rõ ràng)*

---

## 🔧 Troubleshooting

**ValueError: shape mismatch**
```bash
python clean_data.py  # Chạy trước khi analysis
```

**ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

**NLTK stopwords not found**
```bash
python -c "import nltk; nltk.download('stopwords')"
```

---

## 🧪 Tạo Dataset bằng SQLMap (Optional)

Theo phương pháp của bài báo - tạo dataset từ SQLMap:

```bash
# 1. Chạy DVWA
docker run -d --name dvwa -p 8080:80 vulnerables/web-dvwa:1.9

# 2. Generate payloads
sqlmap -u "http://localhost:8080/vulnerabilities/sqli/?id=1&Submit=Submit" \
  --batch --level=2 --risk=1 --technique=BEU -v 3 --stop=50 > sqli_payloads.txt

# 3. Tạo normal.txt với input bình thường
echo -e "id=1\nid=2\nid=admin" > normal.txt

# 4. Chạy build_dataset.py để merge
python build_dataset.py
```

**Chi tiết:** Xem [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)

---

## 📎 Tham khảo

**Paper:**
Casmiry, E., Mduma, N., & Sinde, R. (2025). *Enhanced SQL injection detection using chi-square feature selection and machine learning classifiers.* Frontiers in Big Data. [DOI:10.3389/fdata.2025.1686479](https://doi.org/10.3389/fdata.2025.1686479)

**Dataset:**
SQLiV3 - [Kaggle](https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset)

---

## ✍️ Mở rộng

Ý tưởng phát triển:
- [ ] Implement coarse + fine search tự động
- [ ] So sánh Chi-square vs IG vs MI
- [ ] Thêm Random Forest / XGBoost
- [ ] Deep Learning (LSTM, BERT)
- [ ] Confusion matrix, ROC curves
- [ ] Test trên external datasets