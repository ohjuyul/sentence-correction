import os
import json
from torch.utils.data import Dataset
from transformers import BertTokenizer

class GrammarCorrectionDataset(Dataset):
    def __init__(self, data_dir, tokenizer_name="beomi/kcbert-base", max_length=128):
        self.pairs = []
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

        for fname in files:
            path = os.path.join(data_dir, fname)
            try:
                with open(path, encoding="utf-8-sig") as f:
                    data = json.load(f)

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

        # 강제 범위 제한
        input_ids = source_enc["input_ids"].squeeze(0).clamp(min=0, max=self.tokenizer.vocab_size - 1)
        attention_mask = source_enc["attention_mask"].squeeze(0)
        labels = target_enc["input_ids"].squeeze(0).clamp(min=0, max=self.tokenizer.vocab_size - 1)

        # 패딩 토큰은 -100으로 바꿈
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
