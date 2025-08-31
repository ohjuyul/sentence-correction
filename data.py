# data_txt.py  (Streaming / IterableDataset 버전)
import os
import random
from typing import Dict, Optional, Iterator

import torch
from torch.utils.data import IterableDataset
from transformers import BertTokenizer, PreTrainedTokenizerBase

IGNORE_INDEX = -100


# ------------------------------
# 유틸
# ------------------------------
def ensure_special_tokens(tok: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """디코더 토크나이저에 PAD/BOS/EOS가 없으면 추가."""
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    if getattr(tok, "bos_token", None) is None:
        tok.add_special_tokens({"bos_token": "<s>"})
    if tok.eos_token is None:
        tok.add_special_tokens({"eos_token": "</s>"})
    return tok


def shift_right_1d(labels_1d: torch.Tensor, bos_id: int) -> torch.Tensor:
    """
    (L,) 라벨을 기준으로 디코더 입력을 생성.
    -100(IGNORE_INDEX)은 0으로 치환 후 [BOS, y[:-1]]
    """
    base = labels_1d.masked_fill(labels_1d.eq(IGNORE_INDEX), 0)
    out = base.clone()
    out[1:] = base[:-1]
    out[0] = bos_id
    return out


# ------------------------------
# Streaming Dataset
# ------------------------------
class GrammarCorrectionDataset(IterableDataset):
    """
    대용량(수백만 라인)용 스트리밍 Dataset.
    - 인코더: KC-BERT 토크나이저(입력 원문)
    - 디코더: 사용자 SPM(T5Tokenizer/Fast 등) 토크나이저(정답 문장)
    - 각 샘플을 즉시 토크나이즈 → 배치로 넘기고 메모리에서 버림.

    반환 키:
      input_ids, attention_mask, labels, decoder_input_ids, decoder_attention_mask

    중요:
      - IterableDataset는 DataLoader에서 shuffle=False 사용 권장(필수).
      - stride는 0을 권장(>0이면 예시 수 폭증).
    """
    def __init__(
        self,
        ko_file: str,
        correct_file: str,
        tokenizer_name: str = "beomi/kcbert-base",
        max_length: int = 300,          # 인코더 길이
        stride: int = 0,                # 스트리밍에선 0 권장
        limit: Optional[int] = None,    # 에폭 당 최대 샘플 수
        decoder_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        dec_max_length: int = 512,      # 디코더 길이
        downsample_prob: float = 1.0,   # 0~1, 예: 0.2면 20%만 사용
        seed: int = 42,
    ):
        super().__init__()
        if not (os.path.exists(ko_file) and os.path.exists(correct_file)):
            raise FileNotFoundError(f"ko/correct 파일을 확인하세요: {ko_file}, {correct_file}")

        # 인코더(소스) 토크나이저
        self.src_tok: BertTokenizer = BertTokenizer.from_pretrained(tokenizer_name)

        # 디코더(타깃) 토크나이저 준비
        if decoder_tokenizer is None:
            raise ValueError("decoder_tokenizer를 반드시 전달해야 합니다.")
        self.dec_tok: PreTrainedTokenizerBase = ensure_special_tokens(decoder_tokenizer)
        if getattr(self.dec_tok, "bos_token_id", None) is None:
            raise ValueError("디코더 토크나이저에 bos_token_id가 필요합니다.")
        if self.dec_tok.pad_token_id is None or self.dec_tok.eos_token_id is None:
            raise ValueError("디코더 토크나이저에 pad/eos 토큰이 필요합니다.")

        # 파라미터
        self.ko_file = ko_file
        self.correct_file = correct_file
        self.max_length = max_length
        self.dec_max_length = dec_max_length
        self.stride = stride
        self.limit = limit
        self.downsample_prob = float(downsample_prob)
        self.rng = random.Random(seed)

        if self.stride and self.stride > 0:
            print("[Warn] Streaming에서 stride>0은 예시 수 폭증을 유발합니다. 0을 권장합니다.")

    # -------- 내부 토크나이즈 --------
    def _tok_src(self, text: str) -> Dict[str, torch.Tensor]:
        return self.src_tok(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            stride=0 if self.stride is None else self.stride,
            return_overflowing_tokens=False,  # 스트리밍에선 윈도우 OFF 권장
            return_tensors="pt",
        )

    def _tok_tgt_pack(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.dec_tok(
            text,
            max_length=self.dec_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tgt_ids: torch.Tensor = enc["input_ids"][0]  # (L,)
        pad_id = self.dec_tok.pad_token_id
        bos_id = self.dec_tok.bos_token_id

        labels = tgt_ids.clone()
        labels[labels == pad_id] = IGNORE_INDEX

        dec_inp = shift_right_1d(labels, bos_id)
        dec_attn = (dec_inp != pad_id).long()

        return {
            "labels": labels,                       # (L,)
            "decoder_input_ids": dec_inp,           # (L,)
            "decoder_attention_mask": dec_attn,     # (L,)
        }

    # -------- 필수: 이터레이터 --------
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        produced = 0
        with open(self.ko_file, "r", encoding="utf-8") as f_ko, open(self.correct_file, "r", encoding="utf-8") as f_cor:
            for src, tgt in zip(f_ko, f_cor):
                if self.limit is not None and produced >= self.limit:
                    break
                if self.downsample_prob < 1.0 and self.rng.random() > self.downsample_prob:
                    continue

                src = (src or "").strip()
                tgt = (tgt or "").strip()
                if not src or not tgt:
                    continue

                sp = self._tok_src(src)
                tp = self._tok_tgt_pack(tgt)

                yield {
                    "input_ids": sp["input_ids"][0],                 # (L,)
                    "attention_mask": sp["attention_mask"][0],       # (L,)
                    "labels": tp["labels"],                          # (L,)
                    "decoder_input_ids": tp["decoder_input_ids"],    # (L,)
                    "decoder_attention_mask": tp["decoder_attention_mask"],  # (L,)
                }
                produced += 1

    # IterableDataset에서는 __len__이 필수 아님. progress bar 용으로 limit 반환(있을 때만).
    def __len__(self) -> int:
        return int(self.limit) if self.limit is not None else 0
