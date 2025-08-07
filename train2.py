import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from data2 import GrammarCorrectionDataset
from model2 import CorrectionModel

def shift_tokens_right(input_ids, pad_token_id):
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = pad_token_id
    return shifted

def calculate_accuracy(logits, labels, pad_token_id):
    preds = torch.argmax(logits, dim=-1)
    mask = labels != -100
    correct = (preds == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()

def decode_tokens(tokenizer, token_ids):
    token_ids = token_ids.clone()
    token_ids[token_ids == -100] = tokenizer.pad_token_id
    texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    return texts

def train():
    # GPU가 있으면 GPU 사용, 없으면 CPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_dataset = GrammarCorrectionDataset("../data/train")
    valid_dataset = GrammarCorrectionDataset("../data/valid")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16)

    model = CorrectionModel()
    model.to(device)  # 모델을 GPU 또는 CPU에 올림

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 9
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    tokenizer = model.tokenizer

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        total_loss = 0
        total_acc = 0
        train_steps = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} Train", leave=False)

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 하드코딩 인덱스 제한
            input_ids = input_ids.clamp(min=0, max=model.config.vocab_size - 1)
            labels = torch.where(labels == -100, labels, labels.clamp(min=0, max=model.config.vocab_size - 1))

            decoder_input_ids = shift_tokens_right(labels, pad_token_id=tokenizer.pad_token_id)
            decoder_input_ids = decoder_input_ids.clamp(min=0, max=model.config.vocab_size - 1)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, decoder_input_ids)

            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            acc = calculate_accuracy(logits, labels, tokenizer.pad_token_id)

            total_loss += loss.item()
            total_acc += acc
            train_steps += 1
            loop.set_postfix(loss=loss.item(), accuracy=acc)

        avg_train_loss = total_loss / train_steps
        avg_train_acc = total_acc / train_steps

        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        val_steps = 0
        smoothie = SmoothingFunction().method4

        all_refs = []
        all_hyps = []

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch}/{num_epochs} Valid", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                input_ids = input_ids.clamp(min=0, max=model.config.vocab_size - 1)
                labels = torch.where(labels == -100, labels, labels.clamp(min=0, max=model.config.vocab_size - 1))

                decoder_input_ids = shift_tokens_right(labels, pad_token_id=tokenizer.pad_token_id)
                decoder_input_ids = decoder_input_ids.clamp(min=0, max=model.config.vocab_size - 1)

                logits = model(input_ids, attention_mask, decoder_input_ids)

                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

                acc = calculate_accuracy(logits, labels, tokenizer.pad_token_id)

                val_loss += loss.item()
                val_acc += acc
                val_steps += 1

                pred_tokens = torch.argmax(logits, dim=-1)
                preds_text = decode_tokens(tokenizer, pred_tokens)
                labels_text = decode_tokens(tokenizer, labels)

                for ref, hyp in zip(labels_text, preds_text):
                    ref_tokens = ref.split()
                    hyp_tokens = hyp.split()
                    if len(hyp_tokens) == 0:
                        hyp_tokens = [" "]
                    all_refs.append([ref_tokens])
                    all_hyps.append(hyp_tokens)

        avg_val_loss = val_loss / val_steps
        avg_val_acc = val_acc / val_steps

        bleu_scores = [sentence_bleu(ref, hyp, smoothing_function=smoothie) for ref, hyp in zip(all_refs, all_hyps)]
        avg_bleu = sum(bleu_scores) / len(bleu_scores)

        print(f"Epoch {epoch} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"  Valid Loss: {avg_val_loss:.4f}, Valid Acc: {avg_val_acc:.4f}, BLEU: {avg_bleu:.4f}")

        if epoch % 3 == 0:
            save_path = f"./checkpoints/correction_model_epoch{epoch}.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
