import re
import chardet

input_file = "requirements.txt"
output_file = "cleaned_requirements.txt"

# Auto-detect encoding
raw_data = open(input_file, 'rb').read()
encoding = chardet.detect(raw_data)['encoding']

with open(input_file, "r", encoding=encoding) as f:
    lines = f.readlines()

cleaned = []
seen = set()

for line in lines:
    line = line.strip()

    # Skip empty lines
    if not line:
        continue

    # Skip lines with local file paths (Conda-specific or local builds)
    if "file://" in line or "D:/" in line or "/home/" in line or "/tmp/" in line:
        continue

    # Extract package name for deduplication
    pkg_name = re.split(r"[=<>@]", line)[0].strip().lower()
    if pkg_name in seen:
        continue

    seen.add(pkg_name)
    cleaned.append(line)

# Write cleaned requirements to new file
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned))

print(f"Cleaned requirements written to '{output_file}'")