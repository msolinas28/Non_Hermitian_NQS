import os
import re

pattern = re.compile(r"(.*_k_)([0-9]*\.[0-9]+)(.*)")

for filename in os.listdir():
    match = pattern.match(filename)
    if match:
        prefix, k_str, suffix = match.groups()
        k_val = float(k_str)
        k_rounded = f"{k_val:.3f}"
        new_filename = f"{prefix}{k_rounded}{suffix}"
        if new_filename != filename:
            print(f"Renaming: {filename} → {new_filename}")
            os.rename(filename, new_filename)
