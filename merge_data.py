import pandas as pd

# Load malicious payloads
malicious = pd.read_csv('data/custom_sqli_malicious.csv')
print(f"Generated malicious: {len(malicious)}")

# Load existing dataset (keep only benign)
existing = pd.read_csv('data/SQLiV3_cleaned.csv')
benign = existing[existing['Label'] == 0]
print(f"Existing benign: {len(benign)}")

# Merge
combined = pd.concat([malicious, benign], ignore_index=True)

# Shuffle
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Check balance
print("\n=== FINAL DATASET ===")
print(f"Total: {len(combined)}")
print(f"Malicious (1): {(combined['Label']==1).sum()} ({(combined['Label']==1).sum()/len(combined)*100:.1f}%)")
print(f"Benign (0): {(combined['Label']==0).sum()} ({(combined['Label']==0).sum()/len(combined)*100:.1f}%)")

# Save
combined.to_csv('data/SQLiV3_FULL_65K.csv', index=False)
print(f"\n✅ Saved: data/SQLiV3_FULL_65K.csv")