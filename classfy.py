# extract_pairs.py
from pathlib import Path
import argparse
import json
import sys

# --------------------------- 공통 유틸 ---------------------------

def iter_jsonlike_files(roots):
    """여러 root 아래의 .json/.jsonl 파일을 재귀적으로 순회합니다."""
    for root in roots:
        r = Path(root)
        if not r.exists():
            continue
        for ext in ("*.json", "*.jsonl"):
            yield from r.rglob(ext)

def normalize_line(s):
    """개행/탭/다중 공백을 한 줄로 정규화."""
    if s is None:
        return ""
    return " ".join(str(s).split())

# --------------------------- 파싱 & 추출 ---------------------------

def find_pairs_in_obj(obj):
    """
    객체에서 (ko, corrected) 쌍을 모두 찾아 생성합니다.
    - dict 레벨에서 'ko'와 'corrected'가 함께 있을 때만 한 쌍으로 간주
    - 리스트/중첩 dict는 재귀 탐색
    """
    if isinstance(obj, dict):
        # 같은 dict 레벨에 모두 존재하는 경우에만 페어링
        if "ko" in obj and "corrected" in obj:
            ko = normalize_line(obj.get("ko"))
            corr = normalize_line(obj.get("corrected"))
            if ko and corr:
                yield (ko, corr)
        # 하위로 계속 탐색
        for v in obj.values():
            yield from find_pairs_in_obj(v)
        return
    if isinstance(obj, list):
        for item in obj:
            yield from find_pairs_in_obj(item)
        return
    # 기타 타입은 무시
    return

def parse_json_file(path: Path):
    """파일 하나(.json/.jsonl)에서 (ko, corrected) 쌍을 스트리밍으로 생성."""
    suf = path.suffix.lower()
    if suf == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield from find_pairs_in_obj(obj)
    else:
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                obj = json.load(f)
        except json.JSONDecodeError:
            return
        yield from find_pairs_in_obj(obj)

# --------------------------- 쓰기 ---------------------------

def write_pairs(pairs_iter, ko_out_path: Path, corr_out_path: Path, log_every:int=5000):
    """(ko, corrected) 스트림을 받아 두 파일에 같은 순서로 기록 + 진행 로그."""
    count = 0
    with ko_out_path.open("w", encoding="utf-8") as f_ko, corr_out_path.open("w", encoding="utf-8") as f_corr:
        for ko, corr in pairs_iter:
            f_ko.write(ko + "\n")
            f_corr.write(corr + "\n")
            count += 1
            if log_every and (count % log_every == 0):
                print(f"[pairs] written={count}", file=sys.stderr)
    return count

# --------------------------- 엔트리포인트 ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract (ko, corrected) pairs to txt")
    ap.add_argument("--dirs", nargs="+", required=True, help="검색할 루트 디렉터리들 (예: ./data/train ./data/valid)")
    ap.add_argument("--out-ko", default="./ko.txt")
    ap.add_argument("--out-correct", default="./correct.txt")
    ap.add_argument("--log-every", type=int, default=10000, help="진행 로그 출력 주기(쌍 개수 기준)")
    args = ap.parse_args()

    files_iter = iter_jsonlike_files(args.dirs)

    def gen_pairs_with_progress():
        files_seen = 0
        for p in files_iter:
            files_seen += 1
            if files_seen % 10000 == 0:
                print(f"[scan] files={files_seen}", file=sys.stderr)
            yield from parse_json_file(p)
        print(f"[scan] done. files={files_seen}", file=sys.stderr)

    total_pairs = write_pairs(
        pairs_iter=gen_pairs_with_progress(),
        ko_out_path=Path(args.out_ko),
        corr_out_path=Path(args.out_correct),
        log_every=args.log_every
    )
    print(f"[done] total_pairs={total_pairs}")

if __name__ == "__main__":
    main()
