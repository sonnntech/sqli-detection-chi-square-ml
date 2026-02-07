"""
SCRIPT LÀM SẠCH DỮ LIỆU SQLiV3.csv
====================================

Dataset SQLiV3.csv có thể chứa nhiều labels không mong muốn.
Script này sẽ clean dataset để chỉ giữ lại:
- Label 0: Benign (Normal queries)
- Label 1: Malicious (SQL Injection)

Theo paper: Enhanced SQL injection detection using chi-square feature selection
Section 2.1: Datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("SCRIPT LÀM SẠCH DỮ LIỆU - SQLiV3.csv")
print("=" * 80)

# Load dataset
input_file = Path("data/SQLiV3.csv")

if not input_file.exists():
    print(f"\n❌ Không tìm thấy file: {input_file}")
    print("   Vui lòng đảm bảo file SQLiV3.csv nằm trong cùng thư mục với script này")
    exit(1)

print(f"\n📂 Loading: {input_file}")
df = pd.read_csv(input_file)

print(f"✓ Loaded: {len(df):,} rows, {df.shape[1]} columns")

# Show current state
print(f"\n📊 TRƯỚC KHI CLEAN:")
print(f"   Columns: {list(df.columns)}")
print(f"   Total rows: {len(df):,}")

if 'Label' in df.columns:
    label_counts = df['Label'].value_counts().sort_index()
    print(f"\n   Label distribution:")
    for label, count in label_counts.items():
        print(f"      Label {label}: {count:,} ({count/len(df)*100:.2f}%)")
else:
    print("\n   ⚠️  Không tìm thấy cột 'Label'")

# STEP 1: Keep only necessary columns
print("\n" + "=" * 80)
print("BƯỚC 1: Giữ lại các cột cần thiết")
print("=" * 80)

if 'Sentence' not in df.columns or 'Label' not in df.columns:
    print("❌ Dataset thiếu cột 'Sentence' hoặc 'Label'")
    print(f"   Các cột hiện có: {list(df.columns)}")
    exit(1)

df = df[['Sentence', 'Label']]
print(f"✓ Giữ lại 2 cột: Sentence, Label")

# STEP 2: Handle missing values
print("\n" + "=" * 80)
print("BƯỚC 2: Xử lý missing values")
print("=" * 80)

missing_before = df.isnull().sum().sum()
print(f"Missing values trước: {missing_before}")

df = df.dropna()

missing_after = df.isnull().sum().sum()
print(f"Missing values sau: {missing_after}")
print(f"✓ Đã xóa {missing_before - missing_after} rows có missing values")

# STEP 3: Convert Label to numeric
print("\n" + "=" * 80)
print("BƯỚC 3: Chuyển Label về dạng số")
print("=" * 80)

print(f"Kiểu dữ liệu Label trước: {df['Label'].dtype}")
df['Label'] = pd.to_numeric(df['Label'], errors='coerce')
print(f"Kiểu dữ liệu Label sau: {df['Label'].dtype}")

# Remove rows where Label couldn't be converted
invalid_labels = df['Label'].isna().sum()
if invalid_labels > 0:
    print(f"⚠️  Tìm thấy {invalid_labels} rows có Label không hợp lệ (không thể convert sang số)")
    df = df.dropna(subset=['Label'])
    print(f"✓ Đã xóa {invalid_labels} rows")

df['Label'] = df['Label'].astype(int)

# STEP 4: Keep only Label 0 and 1
print("\n" + "=" * 80)
print("BƯỚC 4: Giữ lại chỉ Label 0 (Benign) và Label 1 (Malicious)")
print("=" * 80)

print(f"\nLabel distribution trước khi filter:")
label_counts_before = df['Label'].value_counts().sort_index()
for label, count in label_counts_before.items():
    print(f"   Label {label}: {count:,}")

# Keep only 0 and 1
df = df[df['Label'].isin([0, 1])]

print(f"\nLabel distribution sau khi filter:")
label_counts_after = df['Label'].value_counts().sort_index()
for label, count in label_counts_after.items():
    label_name = "Benign (Normal)" if label == 0 else "Malicious (SQL Injection)"
    print(f"   Label {label} - {label_name}: {count:,} ({count/len(df)*100:.2f}%)")

rows_removed = len(label_counts_before) - len(label_counts_after)
if rows_removed > 0:
    total_removed = sum(label_counts_before) - sum(label_counts_after)
    print(f"\n✓ Đã xóa {rows_removed} labels khác (tổng {total_removed:,} rows)")

# STEP 5: Remove duplicates
print("\n" + "=" * 80)
print("BƯỚC 5: Xóa các queries trùng lặp")
print("=" * 80)

before_dedup = len(df)
df = df.drop_duplicates(subset=['Sentence'])
after_dedup = len(df)

print(f"Rows trước: {before_dedup:,}")
print(f"Rows sau: {after_dedup:,}")
print(f"✓ Đã xóa {before_dedup - after_dedup:,} duplicates")

# STEP 6: Remove empty queries
print("\n" + "=" * 80)
print("BƯỚC 6: Xóa các queries rỗng hoặc quá ngắn")
print("=" * 80)

before_empty = len(df)
df['query_length'] = df['Sentence'].str.len()
df = df[df['query_length'] > 3]  # Ít nhất 4 ký tự
df = df.drop(columns=['query_length'])
after_empty = len(df)

print(f"✓ Đã xóa {before_empty - after_empty:,} queries rỗng hoặc quá ngắn (<4 chars)")

# FINAL SUMMARY
print("\n" + "=" * 80)
print("📊 TÓNG KẾT SAU KHI CLEAN")
print("=" * 80)

print(f"\n✅ Dataset đã clean:")
print(f"   Total rows: {len(df):,}")
print(f"   Columns: {list(df.columns)}")

label_counts_final = df['Label'].value_counts().sort_index()
print(f"\n   Label distribution:")
for label, count in label_counts_final.items():
    label_name = "Benign" if label == 0 else "Malicious"
    percentage = (count / len(df)) * 100
    print(f"      Label {label} ({label_name}): {count:,} ({percentage:.2f}%)")

# Check balance
ratio = label_counts_final.max() / label_counts_final.min()
print(f"\n   ⚖️  Balance ratio: {ratio:.2f}")
if ratio < 1.5:
    print("      ✓ Dataset cân bằng tốt")
elif ratio < 3:
    print("      ⚠️  Dataset hơi mất cân bằng")
else:
    print("      ❌ Dataset mất cân bằng nghiêm trọng")

# Save cleaned dataset
output_file = "SQLiV3_cleaned.csv"
df.to_csv(output_file, index=False)

print(f"\n💾 ĐÃ LƯU DATASET ĐÃ CLEAN:")
print(f"   File: {output_file}")
print(f"   Size: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")

print("\n" + "=" * 80)
print("✅ HOÀN THÀNH!")
print("=" * 80)
print(f"\n📝 BƯỚC TIẾP THEO:")
print(f"   1. Sử dụng file '{output_file}' cho các phân tích tiếp theo")
print(f"   2. Chạy: python data_analysis.py")
print(f"   3. Hoặc mở Jupyter notebook: data_exploration.ipynb")
