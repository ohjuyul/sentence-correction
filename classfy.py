from pathlib import Path
import json

train_dir = Path("./data/train")
record_count_with_both = 0

def count_records(o):
    if isinstance(o, dict):
        return 1 if "ko" in o and "corrected" in o else 0
    if isinstance(o, list):
        return sum(count_records(item) for item in o)
    return 0

total_files = 0
for path in train_dir.rglob("*.json"):
    total_files += 1
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        continue
    record_count_with_both += count_records(obj)

print(f"train 폴더의 총 JSON 파일 수: {total_files}")
print(f"'ko'와 'corrected' 모두 있는 레코드 수: {record_count_with_both}")
