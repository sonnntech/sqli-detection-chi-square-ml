"""
PHÂN TÍCH CHUYÊN SÂU DỮ LIỆU SQLiV3.csv
========================================

Script này giúp bạn hiểu đầy đủ về:
1. Cấu trúc dữ liệu
2. Đặc trưng của SQL injection
3. Phân phối và patterns
4. Các loại tấn công
5. Đặc điểm ngôn ngữ học (linguistic features)

Tham khảo: Paper Section 2.1 (Datasets)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==============================================================================
# PHẦN 1: LOAD VÀ KHẢO SÁT BAN ĐẦU
# ==============================================================================

def initial_exploration(file_path):
    """
    Bước 1: Khảo sát cơ bản về dataset
    
    Câu hỏi:
    - Dataset có bao nhiêu dòng, cột?
    - Có missing values không?
    - Cột nào là input, cột nào là output?
    - Kiểu dữ liệu ra sao?
    """
    print("=" * 80)
    print("PHẦN 1: KHẢO SÁT CƠ BẢN DATASET")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv(file_path)
    
    print(f"\n📊 Kích thước dataset:")
    print(f"   - Số dòng (samples): {len(df):,}")
    print(f"   - Số cột (features): {df.shape[1]}")
    
    print(f"\n📋 Các cột trong dataset:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col} (dtype: {df[col].dtype})")
    
    print(f"\n🔍 Thông tin chi tiết:")
    print(df.info())
    
    print(f"\n❓ Missing values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✓ Không có missing values")
    else:
        print(missing[missing > 0])
    
    print(f"\n👀 5 dòng đầu tiên:")
    print(df.head())
    
    return df


# ==============================================================================
# PHẦN 2: PHÂN TÍCH NHÃN (LABELS)
# ==============================================================================

def analyze_labels(df):
    """
    Bước 2: Phân tích cột Label
    
    Câu hỏi:
    - Có bao nhiêu class?
    - Phân bố mỗi class như thế nào?
    - Dataset có imbalanced không?
    """
    print("\n" + "=" * 80)
    print("PHẦN 2: PHÂN TÍCH NHÃN (LABELS)")
    print("=" * 80)
    
    # Ensure Label column exists
    if 'Label' not in df.columns:
        print("⚠️  Không tìm thấy cột 'Label'")
        return
    
    print(f"\n📊 Phân bố nhãn:")
    label_counts = df['Label'].value_counts().sort_index()
    
    print(f"\n🔢 Số lượng classes: {len(label_counts)}")
    
    # If more than 2 classes, need to clean first
    if len(label_counts) > 2:
        print(f"\n⚠️  CẢNH BÁO: Dataset có {len(label_counts)} classes!")
        print("   Paper chỉ sử dụng 2 classes: 0 (Benign) và 1 (Malicious)")
        print("   Đang hiển thị tất cả classes hiện tại:")
        print()
        
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   Label {label}: {count:,} ({percentage:.2f}%)")
        
        print("\n💡 GỢI Ý: Bạn cần clean dataset để chỉ giữ lại Label 0 và 1")
        print("   Thêm dòng sau vào code:")
        print("   df = df[df['Label'].isin([0, 1])]")
        
        # Check balance
        ratio = label_counts.max() / label_counts.min()
        print(f"\n⚖️  Class balance ratio: {ratio:.2f}")
        print("   (ratio giữa class lớn nhất và nhỏ nhất)")
        
        # Simple bar chart for all classes
        fig, ax = plt.subplots(figsize=(12, 6))
        
        labels_list = [f"Label {l}" for l in label_counts.index]
        colors_list = plt.cm.Set3(np.linspace(0, 1, len(label_counts)))
        
        bars = ax.bar(labels_list, label_counts.values, color=colors_list, 
                     alpha=0.7, edgecolor='black')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Class Distribution (All Labels)')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, label_counts.values)):
            percentage = (count / len(df)) * 100
            ax.text(bar.get_x() + bar.get_width()/2, count + len(df)*0.01, 
                   f'{count:,}\n({percentage:.1f}%)', 
                   ha='center', fontweight='bold', fontsize=9)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('label_distribution.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: label_distribution.png")
        plt.show()
        
        return
    
    # Original code for binary classification
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        label_name = "Benign (Normal)" if label == 0 else "Malicious (SQL Injection)"
        print(f"   Label {label} - {label_name}:")
        print(f"      Số lượng: {count:,} ({percentage:.2f}%)")
    
    # Check balance
    ratio = label_counts.max() / label_counts.min()
    print(f"\n⚖️  Class balance ratio: {ratio:.2f}")
    if ratio < 1.5:
        print("   ✓ Dataset khá cân bằng (balanced)")
    elif ratio < 3:
        print("   ⚠️  Dataset hơi mất cân bằng (slightly imbalanced)")
    else:
        print("   ❌ Dataset mất cân bằng nghiêm trọng (highly imbalanced)")
    
    return label_counts


# ==============================================================================
# PHẦN 3: PHÂN TÍCH TRUY VẤN (QUERIES)
# ==============================================================================

def analyze_queries(df):
    """
    Bước 3: Phân tích cột Sentence (SQL queries)
    
    Câu hỏi:
    - Độ dài truy vấn như thế nào?
    - Benign vs Malicious có khác biệt về độ dài?
    - Từ nào xuất hiện nhiều nhất?
    """
    print("\n" + "=" * 80)
    print("PHẦN 3: PHÂN TÍCH TRUY VẤN SQL")
    print("=" * 80)
    
    if 'Sentence' not in df.columns:
        print("⚠️  Không tìm thấy cột 'Sentence'")
        return
    
    # Calculate query lengths
    df['query_length'] = df['Sentence'].str.len()
    df['word_count'] = df['Sentence'].str.split().str.len()
    
    print(f"\n📏 Thống kê độ dài truy vấn (ký tự):")
    print(df['query_length'].describe())
    
    print(f"\n📝 Thống kê số từ trong truy vấn:")
    print(df['word_count'].describe())
    
    # Compare benign vs malicious
    print(f"\n🔍 So sánh Benign vs Malicious:")
    
    if 'Label' in df.columns:
        for label in sorted(df['Label'].unique()):
            label_name = "Benign" if label == 0 else "Malicious"
            subset = df[df['Label'] == label]
            
            print(f"\n   {label_name} (Label={label}):")
            print(f"      Độ dài TB: {subset['query_length'].mean():.1f} ký tự")
            print(f"      Độ dài Min: {subset['query_length'].min()}")
            print(f"      Độ dài Max: {subset['query_length'].max()}")
            print(f"      Số từ TB: {subset['word_count'].mean():.1f} từ")
    
    return df


# ==============================================================================
# PHẦN 4: PHÁT HIỆN CÁC LOẠI TẤN CÔNG SQL INJECTION
# ==============================================================================

def detect_attack_types(df):
    """
    Bước 4: Phân loại các loại tấn công SQL injection
    
    Các loại chính (theo paper):
    1. UNION-based: UNION SELECT
    2. Boolean-based (Tautology): OR 1=1, AND 1=1
    3. Time-based blind: SLEEP(), WAITFOR DELAY
    4. Error-based: CAST, CONVERT, Syntax errors
    5. Comment-based: --, #, /* */
    6. Stacked queries: ; (multiple statements)
    """
    print("\n" + "=" * 80)
    print("PHẦN 4: PHÁT HIỆN CÁC LOẠI TẤN CÔNG SQL INJECTION")
    print("=" * 80)
    
    if 'Label' not in df.columns or 'Sentence' not in df.columns:
        print("⚠️  Thiếu cột cần thiết")
        return
    
    # Only analyze malicious queries
    malicious = df[df['Label'] == 1]['Sentence']
    
    print(f"\n🔍 Phân tích {len(malicious):,} malicious queries...")
    
    # Define attack patterns
    attack_patterns = {
        'UNION-based': [
            r'\bUNION\b.*\bSELECT\b',
        ],
        'Boolean-based (Tautology)': [
            r'\bOR\s+[\d\'\"]+\s*=\s*[\d\'\"]+',  # OR 1=1, OR '1'='1'
            r'\bAND\s+[\d\'\"]+\s*=\s*[\d\'\"]+',  # AND 1=1
            r'\bOR\s+1\s*=\s*1\b',
        ],
        'Time-based Blind': [
            r'\bSLEEP\s*\(',
            r'\bWAITFOR\s+DELAY\b',
            r'\bBENCHMARK\s*\(',
            r'\bpg_sleep\s*\(',
        ],
        'Error-based': [
            r'\bCONVERT\s*\(',
            r'\bCAST\s*\(',
            r'\bEXTRACTVALUE\s*\(',
            r'\bUPDATEXML\s*\(',
        ],
        'Comment-based': [
            r'--',
            r'#',
            r'/\*.*?\*/',
        ],
        'Stacked Queries': [
            r';\s*\w+',  # ; followed by another statement
        ],
        'String Manipulation': [
            r'CONCAT\s*\(',
            r'\|\|',  # String concatenation
            r'\+\s*[\'"]',  # String concatenation with +
        ],
    }
    
    # Detect patterns
    attack_counts = {}
    
    for attack_type, patterns in attack_patterns.items():
        count = 0
        for pattern in patterns:
            count += malicious.str.contains(pattern, case=False, regex=True, na=False).sum()
        attack_counts[attack_type] = count
    
    # Display results
    print(f"\n📊 Phân loại các loại tấn công:")
    total_detected = sum(attack_counts.values())
    
    for attack_type, count in sorted(attack_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(malicious)) * 100
        print(f"   {attack_type}:")
        print(f"      Số lượng: {count:,} ({percentage:.2f}% of malicious)")
    
    print(f"\n   ⚠️  Lưu ý: Một query có thể thuộc nhiều loại tấn công")
    print(f"   Total detections: {total_detected:,}")
    
    # Show examples for each type
    print(f"\n📝 VÍ DỤ CỤ THỂ CHO MỖI LOẠI:")
    
    for attack_type, patterns in attack_patterns.items():
        print(f"\n{'='*70}")
        print(f"🔴 {attack_type}")
        print(f"{'='*70}")
        
        # Find examples
        found = False
        for pattern in patterns:
            matches = malicious[malicious.str.contains(pattern, case=False, regex=True, na=False)]
            if len(matches) > 0:
                print(f"\nPattern: {pattern}")
                for i, example in enumerate(matches.head(2), 1):
                    # Truncate long queries
                    display = example[:150] + "..." if len(example) > 150 else example
                    print(f"   Example {i}: {display}")
                found = True
                break
        
        if not found:
            print(f"   (Không tìm thấy ví dụ)")
    
    return attack_counts, len(malicious)


# ==============================================================================
# PHẦN 5: PHÂN TÍCH TỪ VỰNG (VOCABULARY)
# ==============================================================================

def analyze_vocabulary(df):
    """
    Bước 5: Phân tích từ vựng
    
    Câu hỏi:
    - Từ nào xuất hiện nhiều nhất?
    - Có sự khác biệt giữa benign và malicious không?
    - Keywords nào đặc trưng cho SQL injection?
    """
    print("\n" + "=" * 80)
    print("PHẦN 5: PHÂN TÍCH TỪ VỰNG")
    print("=" * 80)
    
    if 'Sentence' not in df.columns or 'Label' not in df.columns:
        return
    
    # Tokenize
    def tokenize(text):
        return re.findall(r'\b\w+\b', str(text).lower())
    
    # Separate benign and malicious
    benign_queries = df[df['Label'] == 0]['Sentence']
    malicious_queries = df[df['Label'] == 1]['Sentence']
    
    # Count words
    benign_words = []
    for query in benign_queries:
        benign_words.extend(tokenize(query))
    
    malicious_words = []
    for query in malicious_queries:
        malicious_words.extend(tokenize(query))
    
    benign_counter = Counter(benign_words)
    malicious_counter = Counter(malicious_words)
    
    print(f"\n📊 Thống kê từ vựng:")
    print(f"   Benign: {len(benign_counter):,} unique words")
    print(f"   Malicious: {len(malicious_counter):,} unique words")
    
    # Top words in each class
    print(f"\n🔝 TOP 20 TỪ PHỔ BIẾN TRONG BENIGN:")
    for i, (word, count) in enumerate(benign_counter.most_common(20), 1):
        print(f"   {i:2d}. {word:15s} : {count:6,} lần")
    
    print(f"\n🔴 TOP 20 TỪ PHỔ BIẾN TRONG MALICIOUS:")
    for i, (word, count) in enumerate(malicious_counter.most_common(20), 1):
        print(f"   {i:2d}. {word:15s} : {count:6,} lần")
    
    # Find discriminative words (chỉ có trong malicious)
    print(f"\n🎯 TỪ ĐẶC TRƯNG CHO SQL INJECTION (chỉ có trong malicious):")
    
    discriminative = []
    for word, count in malicious_counter.most_common(100):
        if word not in benign_counter and count > 10:
            discriminative.append((word, count))
    
    for i, (word, count) in enumerate(discriminative[:30], 1):
        print(f"   {i:2d}. {word:20s} : {count:5,} lần")
    
    return benign_counter, malicious_counter


# ==============================================================================
# PHẦN 6: PHÂN TÍCH KÝ TỰ ĐẶC BIỆT
# ==============================================================================

def analyze_special_characters(df):
    """
    Bước 6: Phân tích ký tự đặc biệt
    
    SQL injection thường chứa các ký tự đặc biệt:
    - Dấu nháy: ', "
    - Comment: --, #, /* */
    - Logic: =, <, >, !=
    - Semicolon: ;
    """
    print("\n" + "=" * 80)
    print("PHẦN 6: PHÂN TÍCH KÝ TỰ ĐẶC BIỆT")
    print("=" * 80)
    
    if 'Sentence' not in df.columns or 'Label' not in df.columns:
        return
    
    special_chars = {
        "Single Quote (')": "'",
        "Double Quote (\")": '"',
        "Semicolon (;)": ";",
        "Double Dash (--)": "--",
        "Hash (#)": "#",
        "Percent (%)": "%",
        "Asterisk (*)": "*",
        "Equals (=)": "=",
        "Pipe (|)": "|",
    }
    
    results = {'Benign': {}, 'Malicious': {}}
    
    for label, label_name in [(0, 'Benign'), (1, 'Malicious')]:
        queries = df[df['Label'] == label]['Sentence']
        
        for char_name, char in special_chars.items():
            count = queries.str.contains(re.escape(char), na=False).sum()
            percentage = (count / len(queries)) * 100
            results[label_name][char_name] = (count, percentage)
    
    # Display
    print(f"\n📊 Tần suất ký tự đặc biệt:")
    print(f"\n{'Character':<20s} | {'Benign':<20s} | {'Malicious':<20s} | Difference")
    print("-" * 80)
    
    for char_name in special_chars.keys():
        benign_count, benign_pct = results['Benign'][char_name]
        mal_count, mal_pct = results['Malicious'][char_name]
        diff = mal_pct - benign_pct
        
        print(f"{char_name:<20s} | {benign_count:5,} ({benign_pct:5.1f}%) | "
              f"{mal_count:5,} ({mal_pct:5.1f}%) | {diff:+6.1f}%")
    
    return results, special_chars


# ==============================================================================
# PHẦN 7: MẪU DỮ LIỆU CỤ THỂ
# ==============================================================================

def show_samples(df, n_samples=5):
    """
    Bước 7: Hiển thị mẫu cụ thể
    
    Giúp hiểu trực quan về dữ liệu
    """
    print("\n" + "=" * 80)
    print("PHẦN 7: MẪU DỮ LIỆU CỤ THỂ")
    print("=" * 80)
    
    if 'Sentence' not in df.columns or 'Label' not in df.columns:
        return
    
    print(f"\n✅ {n_samples} MẪU BENIGN (NORMAL QUERIES):")
    print("=" * 80)
    
    benign_samples = df[df['Label'] == 0].sample(n=n_samples, random_state=42)
    for i, (idx, row) in enumerate(benign_samples.iterrows(), 1):
        print(f"\nSample {i}:")
        print(f"   {row['Sentence']}")
    
    print(f"\n\n🔴 {n_samples} MẪU MALICIOUS (SQL INJECTION):")
    print("=" * 80)
    
    malicious_samples = df[df['Label'] == 1].sample(n=n_samples, random_state=42)
    for i, (idx, row) in enumerate(malicious_samples.iterrows(), 1):
        print(f"\nSample {i}:")
        print(f"   {row['Sentence']}")
        
        # Try to identify attack type
        query = row['Sentence']
        attack_types = []
        
        if re.search(r'\bUNION\b.*\bSELECT\b', query, re.IGNORECASE):
            attack_types.append("UNION-based")
        if re.search(r'\bOR\s+[\d\'\"]+\s*=\s*[\d\'\"]+', query, re.IGNORECASE):
            attack_types.append("Boolean-based")
        if re.search(r'\bSLEEP\s*\(|WAITFOR', query, re.IGNORECASE):
            attack_types.append("Time-based")
        if re.search(r'--|#|/\*', query):
            attack_types.append("Comment-based")
        
        if attack_types:
            print(f"   → Detected: {', '.join(attack_types)}")


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def main():
    """
    Chạy toàn bộ phân tích
    """
    print("\n" + "🔍" * 40)
    print("PHÂN TÍCH CHUYÊN SÂU DỮ LIỆU SQLiV3.csv")
    print("Paper: Enhanced SQL injection detection using chi-square feature selection")
    print("🔍" * 40)
    
    # Check for cleaned file first
    cleaned_file = Path("data/SQLiV3_cleaned.csv")
    original_file = Path("data/SQLiV3.csv")
    
    if cleaned_file.exists():
        print(f"\n✓ Tìm thấy file đã clean: {cleaned_file}")
        file_path = cleaned_file
    elif original_file.exists():
        print(f"\n⚠️  Chỉ tìm thấy file gốc: {original_file}")
        print(f"   Khuyến nghị: chạy 'python clean_data.py' trước")
        print(f"   Đang phân tích file gốc...")
        file_path = original_file
    else:
        print(f"\n❌ Không tìm thấy file SQLiV3.csv hoặc SQLiV3_cleaned.csv")
        print("   Vui lòng đảm bảo file nằm trong cùng thư mục với script này")
        return
    
    # Run all analyses
    df = initial_exploration(file_path)
    
    # Check if dataset needs cleaning
    if 'Label' in df.columns:
        unique_labels = df['Label'].nunique()
        if unique_labels > 2:
            print("\n" + "⚠️" * 40)
            print("CẢNH BÁO: DATASET CẦN CLEAN!")
            print("⚠️" * 40)
            print(f"\n   Dataset có {unique_labels} labels khác nhau")
            print("   Paper chỉ sử dụng 2 labels: 0 (Benign) và 1 (Malicious)")
            print(f"\n💡 GIẢI PHÁP:")
            print(f"   Chạy lệnh sau để clean dataset:")
            print(f"   python clean_data.py")
            print(f"\n   Sau đó chạy lại script này")
            print("\n   Hoặc thêm dòng sau vào code của bạn:")
            print("   df = df[df['Label'].isin([0, 1])]")
            print("\n" + "⚠️" * 40)
    
    label_counts = analyze_labels(df)

    # Only continue if we have binary classification
    if 'Label' not in df.columns or df['Label'].nunique() != 2:
        print("\n⚠️  Phân tích bị dừng do dataset cần clean")
        print("   Vui lòng chạy: python clean_data.py")
        return

    # Run all analyses (console output only)
    df = analyze_queries(df)
    attack_counts, n_malicious = detect_attack_types(df)
    benign_counter, malicious_counter = analyze_vocabulary(df)
    analyze_special_characters(df)
    show_samples(df, n_samples=5)

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "=" * 80)
    print("✅ HOÀN THÀNH PHÂN TÍCH")
    print("=" * 80)


if __name__ == "__main__":
    main()
