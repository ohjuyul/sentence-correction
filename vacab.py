# mix_and_train_spm.py
import random
from pathlib import Path
import sentencepiece as spm

def _read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if s:
                yield s

def make_mixed_corpus(correct_txt: str, ko_txt: str, out_path: str,
                      corrected_ratio: float = 0.90, seed: int = 42, max_lines: int | None = None):
    """
    corrected_ratio: 정답 비율(0.8~0.9 권장)
    max_lines: 전체 학습 줄 수 제한(없으면 자동 계산)
    """
    random.seed(seed)
    cor = list(_read_lines(correct_txt))
    ko  = list(_read_lines(ko_txt))

    if max_lines is None:
        total = int(len(cor) / corrected_ratio)  # 정답 개수 기준으로 전체 계산
        take_cor = len(cor)
    else:
        total = max_lines
        take_cor = int(total * corrected_ratio)

    take_ko = max(0, total - take_cor)

    cor_sample = random.sample(cor, min(take_cor, len(cor)))
    ko_sample  = random.sample(ko,  min(take_ko,  len(ko)))

    pool = cor_sample + ko_sample
    random.shuffle(pool)

    Path(out_path).write_text("\n".join(pool) + "\n", encoding="utf-8")
    print(f"[mix] wrote: {out_path} | total={len(pool)} | corrected={len(cor_sample)} | ko={len(ko_sample)}")

def train_spm_bpe(corpus_txt: str, model_prefix: str = "decoder_spm_bpe24k",
                  vocab_size: int = 24000, num_threads: int = 16):
    spm.SentencePieceTrainer.Train(
        input=corpus_txt,
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=1.0,       # 한글 전체 커버
        input_sentence_size=0,         # 0이면 전체 사용
        shuffle_input_sentence=True,
        normalization_rule_name="nfkc",
        byte_fallback=True,            # 희귀 문자 대응
        hard_vocab_limit=False,        # 약간의 여유 허용
        bos_id=1, eos_id=2, unk_id=0, pad_id=3,
        num_threads=num_threads
    )
    print(f"[spm] saved: {model_prefix}.model, {model_prefix}.vocab")

def quick_eval(sp_model: str, sample_texts: list[str]):
    sp = spm.SentencePieceProcessor(model_file=sp_model)
    print("[eval] vocab_size:", sp.vocab_size(), "bos/eos/unk/pad:", sp.bos_id(), sp.eos_id(), sp.unk_id(), sp.pad_id())
    for t in sample_texts:
        pieces = sp.encode(t, out_type=str)
        ids    = sp.encode(t, out_type=int)
        print("TEXT:", t)
        print("PIECES:", pieces)
        print("IDS:", ids)

if __name__ == "__main__":
    # 1) 코퍼스 섞기 (정답 90% + 원본 10%)
    make_mixed_corpus(correct_txt="correct.txt", ko_txt="ko.txt",
                      out_path="corpus_mixed.txt", corrected_ratio=0.90, seed=42)

    # 2) SPM BPE 24k 학습
    train_spm_bpe("corpus_mixed.txt", model_prefix="decoder_spm_bpe24k", vocab_size=24000, num_threads=16)

    # 3) 간단 점검
    quick_eval("decoder_spm_bpe24k.model", [
        "정선에서 무엇이 가장 좋았는지 알려 줘라.",
        "나는 사과를 먹는 걸 좋아한다.",
    ])
