import csv

payloads = []
with open("data/sqli_payloads.txt") as f:
    for line in f:
        if "[PAYLOAD]" in line:
            payload = line.split("[PAYLOAD]")[-1].strip()
            payloads.append(payload)

normals = []
with open("data/normal.txt") as f:
    for line in f:
        normals.append(line.strip())

with open("data/dataset1.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Sentence", "Label"])

    for n in normals:
        writer.writerow([n, 0])

    for p in payloads:
        writer.writerow([p, 1])

print("dataset1.csv created!")