import os
import json
from torch.utils.data import Dataset
from transformers import BertTokenizer

class GrammarCorrectionDataset(Dataset):
    def __init__(self, data_dir, tokenizer_name="beomi/kcbert-base", max_length=128):
        self.pairs = []
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # data_dir 내 모든 json 파일 리스트
        files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

        for fname in files:
            path = os.path.join(data_dir, fname)
            try:
                with open(path, encoding="utf-8-sig") as f:
                    data = json.load(f)

                    # 데이터가 리스트인지 단일 dict인지 구분
                    if isinstance(data, list):
                        for item in data:
                            src = item.get("ko")
                            tgt = item.get("corrected")
                            if src is not None and tgt is not None:
                                self.pairs.append((src, tgt))
                    else:
                        src = data.get("ko")
                        tgt = data.get("corrected")
                        if src is not None and tgt is not None:
                            self.pairs.append((src, tgt))

            except Exception as e:
                print(f"Error loading {fname}: {e}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source, target = self.pairs[idx]

        source_enc = self.tokenizer(
            source,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # CrossEntropyLoss에 맞게 padding 토큰을 -100으로 바꿈 (loss 계산시 무시됨)
        labels = target_enc["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_enc["input_ids"].squeeze(0),
            "attention_mask": source_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }
