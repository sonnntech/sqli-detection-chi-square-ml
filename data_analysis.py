import pandas as pd
import re
from pathlib import Path
from collections import Counter
import urllib.parse

# =========================
# Helpers
# =========================

def normalize_payload(text):
    """Decode URL encoding + remove obfuscation"""
    try:
        text = urllib.parse.unquote(str(text))
        text = text.replace('/**/', ' ')
        return text
    except:
        return text


def tokenize(text):
    """Clean tokenization for vocabulary analysis"""
    text = normalize_payload(text.lower())
    return re.findall(r'\b[a-zA-Z_]+\b', text)


# =========================
# Basic Dataset Info
# =========================

def basic_info(df):
    print("\n" + "="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    print(df.info())

    print("\nLabel Distribution:")
    print(df['Label'].value_counts(normalize=True) * 100)

    print("\nSample rows:")
    print(df.head())


# =========================
# Length Analysis
# =========================

def analyze_length(df):
    print("\n" + "="*80)
    print("QUERY LENGTH ANALYSIS")
    print("="*80)

    df['length'] = df['Sentence'].apply(len)

    benign = df[df['Label'] == 0]['length']
    malicious = df[df['Label'] == 1]['length']

    print(f"Avg benign length: {benign.mean():.2f}")
    print(f"Avg malicious length: {malicious.mean():.2f}")


# =========================
# Attack Pattern Detection
# =========================

def detect_attack_types(df):
    print("\n" + "="*80)
    print("ATTACK TYPE ANALYSIS")
    print("="*80)

    malicious = df[df['Label'] == 1]['Sentence'].apply(normalize_payload)

    patterns = {
        'Comment-based': r'(--|#|/\*)',
        'Boolean-based': r'\b(or|and)\s+1=1',
        'UNION-based': r'\bunion\b',
        'Time-based': r'(sleep|waitfor|pg_sleep)',
        'Error-based': r'(convert|cast|extractvalue|updatexml)',
        'Stacked-query': r';'
    }

    for name, pattern in patterns.items():
        count = malicious.str.contains(pattern, case=False, regex=True).sum()
        pct = (count / len(malicious)) * 100
        print(f"{name:<15}: {pct:6.2f}% ({count})")


# =========================
# Vocabulary Analysis
# =========================

def analyze_vocabulary(df):
    print("\n" + "="*80)
    print("TOP WORDS ANALYSIS")
    print("="*80)

    benign = df[df['Label'] == 0]['Sentence']
    malicious = df[df['Label'] == 1]['Sentence']

    benign_tokens = []
    malicious_tokens = []

    for s in benign:
        benign_tokens.extend(tokenize(s))

    for s in malicious:
        malicious_tokens.extend(tokenize(s))

    print("\nTop benign words:")
    print(Counter(benign_tokens).most_common(15))

    print("\nTop malicious words:")
    print(Counter(malicious_tokens).most_common(15))


# =========================
# Generator Impact Analysis
# =========================

def analyze_generator_effect(df):
    print("\n" + "="*80)
    print("PAYLOAD GENERATOR IMPACT")
    print("="*80)

    df['is_encoded'] = df['Sentence'].str.contains('%[0-9a-fA-F]{2}', regex=True)
    df['has_obfuscation'] = df['Sentence'].str.contains(r'/\*\*/', regex=True)

    encoded_pct = df['is_encoded'].mean() * 100
    obf_pct = df['has_obfuscation'].mean() * 100

    print(f"Payload có URL encoding: {encoded_pct:.2f}%")
    print(f"Payload có obfuscation /**/: {obf_pct:.2f}%")


# =========================
# MAIN
# =========================

def main():
    print("SQLi Dataset Analysis")

    full_file = Path("data/SQLiV3_FULL_65K.csv")
    cleaned_file = Path("data/SQLiV3_cleaned.csv")
    original_file = Path("data/SQLiV3.csv")

    if full_file.exists():
        print(f"\n✓ Using FULL dataset: {full_file}")
        file_path = full_file
    elif cleaned_file.exists():
        print(f"\n✓ Using cleaned dataset: {cleaned_file}")
        file_path = cleaned_file
    elif original_file.exists():
        print(f"\n⚠ Using original dataset: {original_file}")
        file_path = original_file
    else:
        print("❌ No dataset found.")
        return

    df = pd.read_csv(file_path)

    basic_info(df)
    analyze_length(df)
    detect_attack_types(df)
    analyze_vocabulary(df)
    analyze_generator_effect(df)


if __name__ == "__main__":
    main()