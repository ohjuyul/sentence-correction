# spm_train_with_validation.py
import os
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import sentencepiece as spm


# ------------------------------
# 공통 유틸
# ------------------------------
def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def write_lines(path: str, lines: List[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")


def split_train_valid(lines: List[str], valid_ratio: float = 0.1, seed: int = 42) -> Tuple[List[str], List[str]]:
    random.seed(seed)
    idx = list(range(len(lines)))
    random.shuffle(idx)
    cut = int(len(idx) * (1.0 - valid_ratio))
    train_idx, valid_idx = idx[:cut], idx[cut:]
    train = [lines[i] for i in train_idx]
    valid = [lines[i] for i in valid_idx]
    return train, valid


# ------------------------------
# 혼합 코퍼스(정답 80~90% + 원본 10~20%)
# ------------------------------
def make_mixed_corpus_from_lists(
    corrected: List[str],
    originals: List[str],
    corrected_ratio: float = 0.90,
    seed: int = 42,
) -> List[str]:
    """
    corrected_ratio: 정답 비율 (0.80~0.90 권장)
    """
    random.seed(seed)
    total = int(len(corrected) / corrected_ratio) if corrected_ratio > 0 else len(corrected)
    take_cor = min(len(corrected), int(total * corrected_ratio))  # 안전
    take_ko = max(0, total - take_cor)
    cor_sample = random.sample(corrected, take_cor)
    ko_sample = random.sample(originals, min(take_ko, len(originals)))
    pool = cor_sample + ko_sample
    random.shuffle(pool)
    return pool


# ------------------------------
# SentencePiece 학습/로드/평가
# ------------------------------
def train_sentencepiece_bpe(
    corpus_txt: str,
    model_prefix: str = "decoder_spm_bpe24k",
    vocab_size: int = 24000,
    num_threads: int = 16,
) -> Tuple[str, str]:
    spm.SentencePieceTrainer.Train(
        input=corpus_txt,
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=1.0,           # 한글
        input_sentence_size=0,            # 전체 사용
        shuffle_input_sentence=True,
        normalization_rule_name="nfkc",
        byte_fallback=True,               # 희귀 문자 안전망
        hard_vocab_limit=False,           # 약간 여유 허용
        bos_id=1, eos_id=2, unk_id=0, pad_id=3,
        num_threads=num_threads,
    )
    return f"{model_prefix}.model", f"{model_prefix}.vocab"


def check_unk_ratio(sp_model_path: str, lines: List[str], sample_size: int | None = None) -> Tuple[float, float]:
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    unk_id = sp.unk_id()
    total_tokens = 0
    unk_tokens = 0
    total_lines = 0

    iterable = lines if sample_size is None else lines[:sample_size]
    for s in iterable:
        ids = sp.encode(s, out_type=int)
        total_tokens += len(ids)
        unk_tokens += sum(1 for tid in ids if tid == unk_id)
        total_lines += 1

    unk_ratio = (unk_tokens / total_tokens) if total_tokens else 0.0
    avg_len = (total_tokens / total_lines) if total_lines else 0.0
    return unk_ratio, avg_len


# ------------------------------
# 메인 파이프라인
# ------------------------------
def run_pipeline(
    ko_txt: str = "ko.txt",
    correct_txt: str = "correct.txt",
    out_dir: str = "spm_out",
    corrected_ratio: float = 0.90,   # 0.80~0.90 권장
    vocab_size: int = 24000,         # 필요 시 32000
    valid_ratio: float = 0.10,       # 10% 검증
    seed: int = 42,
    num_threads: int = 16,
    sample_eval: int | None = 50000, # UNK 체크에 사용할 검증 줄 수(없으면 전체)
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1) 원본/정답 로드
    originals = read_lines(ko_txt)
    corrected = read_lines(correct_txt)
    assert len(originals) > 0 and len(corrected) > 0, "ko.txt / correct.txt 둘 다 비어있지 않아야 합니다."

    # 2) 훈련/검증 분리 (정답, 원본 각각 분리)
    cor_train, cor_valid = split_train_valid(corrected, valid_ratio=valid_ratio, seed=seed)
    ko_train,  ko_valid  = split_train_valid(originals, valid_ratio=valid_ratio, seed=seed)

    # 3) 혼합 코퍼스 생성 (훈련용: 정답 90% + 원본 10%)
    mixed_train = make_mixed_corpus_from_lists(cor_train, ko_train, corrected_ratio=corrected_ratio, seed=seed)

    # (선택) 검증용 혼합도 만들 수 있지만, 보통 검증은 "정답셋"으로 보는 게 타당
    write_lines(Path(out_dir, "train_mixed.txt").as_posix(), mixed_train)
    write_lines(Path(out_dir, "valid_correct.txt").as_posix(), cor_valid)  # 검증은 정답 문장 위주 권장

    # 4) SPM 학습
    model_prefix = Path(out_dir, f"decoder_spm_bpe{vocab_size//1000}k").as_posix()
    model_file, vocab_file = train_sentencepiece_bpe(
        corpus_txt=Path(out_dir, "train_mixed.txt").as_posix(),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        num_threads=num_threads,
    )
    print(f"[SPM] saved: {model_file}, {vocab_file}")

    # 5) UNK/평균 토큰 길이 평가 (검증 정답셋 기준)
    valid_lines = read_lines(Path(out_dir, "valid_correct.txt").as_posix())
    unk_ratio, avg_len = check_unk_ratio(model_file, valid_lines, sample_size=sample_eval)
    print(f"[EVAL] valid UNK ratio: {unk_ratio*100:.3f}% | avg tokens/line: {avg_len:.2f}")

    # (선택) 훈련셋에서도 확인
    train_unk, train_avg = check_unk_ratio(model_file, mixed_train, sample_size=sample_eval)
    print(f"[EVAL] train UNK ratio: {train_unk*100:.3f}% | avg tokens/line: {train_avg:.2f}")


if __name__ == "__main__":
    # 기본값으로 실행 (파일명/비율/크기 자유 변경)
    run_pipeline(
        ko_txt="ko.txt",
        correct_txt="correct.txt",   # 파일명이 'correcet.txt'라면 그대로 바꿔 넣으세요.
        out_dir="spm_out",
        corrected_ratio=0.90,        # 0.80~0.90 사이로 실험
        vocab_size=24000,            # 필요시 32000으로 재학습
        valid_ratio=0.10,
        seed=42,
        num_threads=16,
        sample_eval=50000,
    )
