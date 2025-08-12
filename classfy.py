# extract_texts_here.py
from pathlib import Path
import json

def iter_jsonlike_files(root: Path):
    for ext in ("*.json", "*.jsonl"):
        yield from root.rglob(ext)

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).split())

def yield_pairs_from_json_obj(obj):
    if isinstance(obj, dict) and ("ko" in obj or "corrected" in obj):
        ko = normalize_text(obj.get("ko", ""))
        corrected = normalize_text(obj.get("corrected", ""))
        if ko and corrected:
            yield (ko, corrected)
        return
    if isinstance(obj, list):
        for item in obj:
            yield from yield_pairs_from_json_obj(item)
        return

def parse_json_file(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield from yield_pairs_from_json_obj(obj)
    else:
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except json.JSONDecodeError:
            return
        yield from yield_pairs_from_json_obj(obj)

def write_pairs(pairs_iter, ko_out: Path, correct_out: Path):
    with ko_out.open("w", encoding="utf-8") as f_ko, correct_out.open("w", encoding="utf-8") as f_corr:
        for ko, corr in pairs_iter:
            f_ko.write(ko + "\n")
            f_corr.write(corr + "\n")

def main():
    root_dirs = [Path("./data/train"), Path("./data/valid")]
    def gen_pairs():
        for root in root_dirs:
            if root.exists():
                for p in iter_jsonlike_files(root):
                    yield from parse_json_file(p)

    write_pairs(gen_pairs(), Path("./ko.txt"), Path("./correct.txt"))

if __name__ == "__main__":
    main()
