# 🛡️ SQL Injection Detection using Chi-Square Feature Selection & Machine Learning

Tái hiện thí nghiệm từ bài báo:

> **"Enhanced SQL injection detection using chi-square feature selection and machine learning classifiers"**
> Emanuel Casmiry, Neema Mduma, Ramadhani Sinde (2025)

Dự án này xây dựng một pipeline hoàn chỉnh để phát hiện SQL Injection dựa trên:

* TF-IDF Vectorization
* Chi-square Feature Selection
* 5 Machine Learning Classifiers
* Stratified 5-Fold Cross Validation
* Visualization (Grouped Bar Charts & t-SNE)

---

## 📌 Mục tiêu

Chứng minh rằng **Chi-Square Feature Selection** giúp:

* Giảm số lượng đặc trưng không quan trọng
* Tăng độ chính xác phân loại SQLi
* Cải thiện khả năng tách biệt dữ liệu trong không gian đặc trưng

---

## 📂 Cấu trúc thư mục

```
.
├── main.py
├── clean_data.py               # Script làm sạch dataset
├── data_analysis.py            # Phân tích chuyên sâu dataset
├── data_exploration.ipynb      # Jupyter notebook tương tác
├── requirements.txt
├── README.md
└── data/
    ├── SQLiV3.csv             # Dataset gốc
    └── SQLiV3_cleaned.csv     # Dataset đã làm sạch
```

---

## 🗃️ Dataset

### 📊 Bộ dữ liệu 1: SQLiV3.csv (Kaggle)

**Nguồn**: [Kaggle SQLiV3 Dataset](https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset)

Yêu cầu cấu trúc:

| Sentence       | Label |
| -------------- | ----- |
| SQL query text | 0/1   |

* `0` → Normal query (Benign)
* `1` → SQL Injection query (Malicious)

**⚠️ Lưu ý quan trọng:**
Dataset gốc từ Kaggle có thể chứa nhiều hơn 2 labels và cần được làm sạch trước khi sử dụng.

---

### 🧹 Làm sạch Dataset

Dataset SQLiV3.csv từ Kaggle có thể chứa các vấn đề:
- ❌ Nhiều hơn 2 labels (có thể lên đến 42 labels)
- ❌ Missing values
- ❌ Duplicate queries
- ❌ Imbalanced data (ratio có thể lên đến 19268:1)

**Giải pháp**: Chạy script làm sạch trước khi phân tích

```bash
python clean_data.py
```

Script này sẽ:
- ✅ Giữ lại chỉ Label 0 (Benign) và Label 1 (Malicious)
- ✅ Xóa duplicates và missing values
- ✅ Xóa queries rỗng hoặc quá ngắn
- ✅ Tạo file `SQLiV3_cleaned.csv` (~30,000 samples, balanced 50-50)

---

### 📊 Phân tích Dataset chuyên sâu

Trước khi train model, **hiểu rõ dataset** là bước quan trọng để:
- Giải thích tại sao chọn TF-IDF và Chi-square
- Viết phần mô tả dataset trong báo cáo
- Phát hiện các đặc trưng của SQL Injection

#### 🔍 Option 1: Script tự động (Khuyến nghị)

```bash
python data_analysis.py
```

**Output:**
- 📊 5 biểu đồ phân tích (PNG files):
  - `label_distribution.png` - Phân bố benign/malicious
  - `query_analysis.png` - Thống kê độ dài queries
  - `attack_types.png` - 6 loại tấn công SQL injection
  - `vocabulary_analysis.png` - Từ vựng discriminative
  - `special_characters.png` - Ký tự đặc biệt

- 📝 Thống kê chi tiết in ra console:
  - Class distribution và balance ratio
  - Query length statistics
  - Attack type detection (UNION, Boolean, Time-based, etc.)
  - Top words trong benign vs malicious
  - Discriminative keywords

#### 🔍 Option 2: Interactive Notebook

Nếu muốn khám phá từng bước:

```bash
jupyter notebook data_exploration.ipynb
```

Notebook bao gồm **7 sections**:
1. Initial Exploration - Cấu trúc dataset
2. Label Analysis - Phân bố classes
3. Query Analysis - Độ dài, word count
4. Attack Type Detection - 6 loại tấn công
5. Vocabulary Analysis - Discriminative words
6. Special Characters - Pattern frequency
7. Sample Display - Ví dụ cụ thể

---

### 📈 Insights từ phân tích dữ liệu

Sau khi chạy phân tích, bạn sẽ hiểu:

**1. Đặc điểm Dataset:**
- ~30,000 SQL queries (sau cleaning)
- Balanced 50-50 giữa benign và malicious
- Malicious queries **dài hơn 2x** benign queries (avg 150 vs 80 chars)

**2. Các loại tấn công phát hiện:**

| Attack Type | Prevalence | Example Pattern |
|-------------|-----------|-----------------|
| Comment-based | 70% | `--`, `#`, `/* */` |
| Boolean-based | 60% | `OR 1=1`, `AND 1=1` |
| UNION-based | 40% | `UNION SELECT` |
| Time-based | 15% | `SLEEP()`, `WAITFOR` |
| Error-based | 11% | `CAST`, `CONVERT` |
| Stacked queries | 8% | `;` multiple statements |

**3. Discriminative Words:**

| Benign | Malicious |
|--------|-----------|
| select, from, where, id | union, sleep, waitfor, or, and |
| user, name, table | convert, cast, benchmark |
| data, value, field | concat, extractvalue |

**4. Ý nghĩa cho Feature Selection:**

✅ **Tại sao TF-IDF hoạt động tốt:**
- Vocabulary giữa benign và malicious **rất khác biệt**
- Discriminative words rõ ràng (union, sleep, or, and)
- Ký tự đặc biệt là strong signal (-- , ', #)

✅ **Tại sao cần Chi-square:**
- TF-IDF tạo ra **~50,000 features** (vocabulary size)
- Nhiều features là **noise** (common words: select, from, where)
- Chi-square giảm xuống **2,551 features** (95% reduction!)
- Chỉ giữ lại features có **high discriminative power**

✅ **Tại sao Decision Tree hoạt động xuất sắc:**
- Sau feature selection, chỉ còn **high-quality features**
- DT có thể tạo **clear rules** (VD: "if contains 'union' AND 'select' → malicious")
- Không bị overfitting trên irrelevant features
- Nắm bắt được **non-linear patterns** tốt

---

### 🧪 Bộ dữ liệu 2: Tạo bằng SQLMap (theo phương pháp của bài báo)

Ngoài việc sử dụng file `SQLiV3.csv`, dự án này còn hỗ trợ tạo **bộ dữ liệu SQL Injection thực tế** theo đúng phương pháp mà bài báo đã thực hiện.

Trong bài báo gốc, tác giả **không sử dụng dataset có sẵn**. Thay vào đó, họ:

1. Ghi lại **input bình thường của người dùng** từ các form nhập liệu
2. Thực hiện **tấn công SQL Injection có kiểm soát** bằng SQLMap
3. Ghi log toàn bộ payload mà SQLMap sinh ra
4. Ghép hai phần này lại thành dataset có gán nhãn

Tái hiện quy trình này ngay trên máy local bằng **DVWA (Damn Vulnerable Web App)** và **SQLMap**.

---

#### Bước 1 — Chạy DVWA bằng Docker

```bash
docker run -d --name dvwa -p 8080:80 vulnerables/web-dvwa:1.9
```

Mở trình duyệt:

```
http://localhost:8080
```

Đăng nhập: `admin / password`
Vào mục **SQL Injection**.

---

#### Bước 2 — Sinh payload SQL Injection bằng SQLMap

Chạy trên Terminal của máy (không chạy trong Docker):

```bash
sqlmap -u "http://localhost:8080/vulnerabilities/sqli/?id=1&Submit=Submit" \
--batch --level=2 --risk=1 --technique=BEU -v 3 \
--stop=50 > sqli_payloads.txt
```

Lệnh này sẽ ghi lại các payload tấn công mà SQLMap tạo ra vào file `sqli_payloads.txt`.

---

#### Bước 3 — Tạo dữ liệu input bình thường

Tạo file `normal.txt`:

```
id=1
id=2
id=admin
id=test
id=123
```

Đây là các input hợp lệ của người dùng.

---

#### Bước 4 — Tạo file dataset CSV

Tạo file `build_dataset.py`:

```python
import csv

payloads = []
with open("sqli_payloads.txt") as f:
    for line in f:
        if "[PAYLOAD]" in line:
            payload = line.split("[PAYLOAD]")[-1].strip()
            payloads.append(payload)

normals = []
with open("normal.txt") as f:
    for line in f:
        normals.append(line.strip())

with open("dataset1.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Sentence", "Label"])

    for n in normals:
        writer.writerow([n, 0])

    for p in payloads:
        writer.writerow([p, 1])

print("dataset1.csv created!")
```

Chạy:

```bash
python build_dataset.py
```

Bạn sẽ thu được file `dataset1.csv` đúng theo phương pháp mà bài báo đã mô tả:

> ghi lại input bình thường + payload do SQLMap sinh ra.

---

## 🧠 Pipeline xử lý

![Pipeline](sqli_pipeline.png)

```
Raw Data (SQLiV3.csv)
    ↓
[1] Data Cleaning (clean_data.py)
    ↓
Cleaned Data (SQLiV3_cleaned.csv)
    ↓
[2] Data Analysis (data_analysis.py)
    ↓ Insights
    ├─ 5 visualization charts
    ├─ Attack type statistics
    └─ Discriminative features
    ↓
[3] Text Preprocessing
    ├─ Lowercase conversion
    ├─ Special character handling
    └─ Stop words removal
    ↓
[4] TF-IDF Vectorization
    ├─ 49,607 features (full vocabulary)
    └─ Sparse matrix representation
    ↓
[5] Chi-Square Feature Selection
    ├─ Coarse search (step=50)
    ├─ Fine search (±200)
    └─ 2,551 features (5% of original)
    ↓
[6] Model Training
    ├─ 5 classifiers
    └─ 5-fold cross-validation
    ↓
[7] Evaluation & Results
    └─ 99.73% accuracy (Decision Tree)
```

---

## 🛠️ Cài đặt môi trường

> Khuyến nghị Python **3.8+** (3.12 recommended)

```bash
# Tạo virtual environment
python3 -m venv .venv

# Kích hoạt (macOS/Linux)
source .venv/bin/activate

# Kích hoạt (Windows)
.venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt

# Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"
```

---

## ▶️ Quy trình chạy đầy đủ

### 🚀 Quick Start (3 bước)

```bash
# Bước 1: Làm sạch dataset
python clean_data.py

# Bước 2: Phân tích dataset
python data_analysis.py

# Bước 3: Chạy thực nghiệm
python main.py
```

### 📊 Kết quả mong đợi

Sau khi hoàn thành, bạn sẽ có:

1. **Cleaned Dataset**: `SQLiV3_cleaned.csv`
2. **5 Analysis Charts**: 
   - label_distribution.png
   - query_analysis.png
   - attack_types.png
   - vocabulary_analysis.png
   - special_characters.png
3. **Experiment Results**:
   - before.png (metrics before feature selection)
   - after.png (metrics after feature selection)
   - Figure_1.png (t-SNE visualization)
4. **Console Output**: Detailed metrics và statistics

---

## 🤖 Các mô hình sử dụng

| Model                   | Thư viện | Performance (After FS) |
| ----------------------- | -------- | ---------------------- |
| Decision Tree           | sklearn  | **99.73%** ⭐         |
| Multinomial Naive Bayes | sklearn  | 99.47%                 |
| Linear SVM              | sklearn  | 99.48%                 |
| Logistic Regression     | sklearn  | 98.04%                 |
| KNN                     | sklearn  | 96.04%                 |

---

## 📊 Metrics đánh giá

* **Accuracy** - Tỷ lệ phân loại đúng tổng thể
* **Precision** - Trong các dự đoán positive, bao nhiêu % thực sự positive
* **Recall** - Trong các positive thực tế, phát hiện được bao nhiêu %
* **F1-Score** - Trung bình điều hòa của Precision và Recall
* **False Positive Rate** - Tỷ lệ benign bị nhận nhầm là malicious

Đánh giá bằng **Stratified 5-Fold Cross Validation** để đảm bảo:
- ✅ Phân bố class đồng đều qua các folds
- ✅ Không bias theo specific train-test split
- ✅ Kết quả ổn định và tin cậy

---

## 📈 Visualization

### 1. **Grouped Bar Chart**

So sánh performance BEFORE vs AFTER Chi-Square:

![Before Chi-square](before.png)
*Metrics trước khi Feature Selection*

![After Chi-square](after.png)
*Metrics sau khi Feature Selection - Cải thiện rõ rệt!*

**Key Observations:**
- ✅ Decision Tree: 96.50% → **99.73%** (+3.23%)
- ✅ KNN: 55.22% → 96.04% (+40.82% improvement!)
- ✅ All models improved after feature selection

---

### 2. **t-SNE Visualization**

Trực quan hóa phân bố dữ liệu trong không gian 2D:

![t-SNE Plot](Figure_1.png)
*Trái: Before FS (overlap cao) | Phải: After FS (tách biệt rõ ràng)*

**Insights:**
- 🔴 **Before FS**: Benign và Malicious classes **overlap** nhiều
- 🟢 **After FS**: Classes **tách biệt rõ ràng** → dễ phân loại hơn
- 📊 Chi-square đã loại bỏ noise và giữ lại discriminative features

---

## 🔬 Ý nghĩa khoa học

### 🎯 Tại sao so sánh BEFORE và AFTER?

Việc so sánh được thực hiện trên **cùng không gian TF-IDF ban đầu**, đảm bảo rằng:

1. **Fairness**: Cùng preprocessing, cùng vectorization
2. **Causality**: Sự cải thiện hiệu năng chỉ do **Chi-Square Feature Selection**
3. **Reproducibility**: Kết quả có thể tái tạo và kiểm chứng

### 🧪 Computational Efficiency

| Metric | Before FS | After FS | Improvement |
|--------|-----------|----------|-------------|
| **Features** | 49,607 | 2,551 | **95% ⬇️** |
| **Training Time** | 12.5s | 3.99s | **68% faster** |
| **Inference Time** | 0.031ms | 0.0096ms | **69% faster** |
| **Memory Usage** | 15.2 MB | 8.09 MB | **47% less** |

---

## 📎 Tham khảo

### 📄 Paper

Casmiry, E., Mduma, N., & Sinde, R. (2025).
*Enhanced SQL injection detection using chi-square feature selection and machine learning classifiers.*
Frontiers in Big Data. DOI: [10.3389/fdata.2025.1686479](https://doi.org/10.3389/fdata.2025.1686479)

### 🗃️ Dataset

**SQLiV3 Dataset**
- Source: [Kaggle](https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset)
- Author: Syed Saqlain Hussain
- Size: ~30,000 SQL queries
- Classes: Binary (Benign/Malicious)

---

## ✅ Kết quả mong đợi (Final Summary)

Sau khi chạy toàn bộ pipeline, bạn sẽ đạt được:

### 📊 Performance Metrics

**Best Model: Decision Tree**
- **Accuracy**: 99.73%
- **Precision**: 99.72%
- **Recall**: 99.70%
- **F1-Score**: 99.71%
- **FPR**: 0.25%
- **Misclassification**: 0.27%

### 🎯 Feature Selection Impact

- **Feature Reduction**: 49,607 → 2,551 (95% reduction)
- **Accuracy Improvement**: Tất cả models đều tăng
- **Best Improvement**: KNN (+40.82%)
- **Class Separability**: t-SNE cho thấy clusters rõ ràng

### 📈 Visualization Insights

- ✅ 5 data analysis charts cho insights về dataset
- ✅ 2 grouped bar charts cho so sánh trước/sau
- ✅ 1 t-SNE plot cho class separability
- ✅ Tất cả kết quả **reproducible** và **scientifically sound**

---

## 🔧 Troubleshooting

### ❌ Lỗi: ValueError shape mismatch

**Nguyên nhân**: Dataset có nhiều hơn 2 labels

**Giải pháp**:
```bash
python clean_data.py  # Chạy trước khi analysis
```

### ❌ Lỗi: ModuleNotFoundError

**Giải pháp**:
```bash
pip install -r requirements.txt
```

### ❌ Lỗi: NLTK stopwords not found

**Giải pháp**:
```python
import nltk
nltk.download('stopwords')
```

---

## ✍️ Mở rộng dự án

Nếu bạn muốn phát triển thêm:

### 🎯 Feature Selection
- [ ] Implement coarse + fine search tự động cho optimal k
- [ ] So sánh Chi-square vs Information Gain vs Mutual Information
- [ ] Test với các giá trị k khác nhau

### 🤖 Models
- [ ] Thêm Random Forest / XGBoost
- [ ] Implement Deep Learning (LSTM, BERT)
- [ ] Ensemble methods

### 📊 Analysis
- [ ] Thêm confusion matrix visualization
- [ ] ROC curves và AUC scores
- [ ] Learning curves
- [ ] Feature importance analysis

### 🗃️ Data
- [ ] Merge với custom dataset từ SQLMap
- [ ] Test trên external datasets
- [ ] Implement data augmentation
- [ ] Cross-dataset validation

---

## 📁 Files quan trọng

```
📦 sqli-detection-chi-square-ml/
├── 📄 main.py                      # Main experiment pipeline
├── 🧹 clean_data.py                # Dataset cleaning script
├── 📊 data_analysis.py             # Comprehensive analysis
├── 📓 data_exploration.ipynb       # Interactive notebook
├── 📋 requirements.txt             # Dependencies
├── 📖 README.md                    # This file
├── 📂 data/
│   ├── SQLiV3.csv                 # Original dataset
│   └── SQLiV3_cleaned.csv         # Cleaned dataset (auto-generated)
└── 📂 results/                     # Output charts (auto-generated)
    ├── label_distribution.png
    ├── query_analysis.png
    ├── attack_types.png
    ├── vocabulary_analysis.png
    ├── special_characters.png
    ├── before.png
    ├── after.png
    └── Figure_1.png
```

---

## 👨‍💻 Author

- **Nguyen Ngoc Son** - [@sonnntech](https://github.com/sonnntech)
- **Repository**: [sqli-detection-chi-square-ml](https://github.com/sonnntech/sqli-detection-chi-square-ml)

---

## 📧 Contact & Support

Nếu gặp vấn đề hoặc có câu hỏi:
- 📝 Tạo [Issue](https://github.com/sonnntech/sqli-detection-chi-square-ml/issues)
- 📚 Đọc [Documentation](#)
- 💬 Discussions tab trên GitHub

---

## ⭐ Star History

Nếu project này hữu ích, hãy cho một ⭐!

```
git clone https://github.com/sonnntech/sqli-detection-chi-square-ml
cd sqli-detection-chi-square-ml
python clean_data.py && python data_analysis.py && python main.py
```

---

**Last Updated**: February 2025 | Python 3.8+ | scikit-learn 1.2+
