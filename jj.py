import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
import sentencepiece as spm
import numpy as np
import random
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 1. Load Teacher Model from raw checkpoint
def load_teacher_model(model_path, tokenizer_path, params_path, device='cuda'):
    """
    Load the teacher model and tokenizer (GPT-2 for testing).
    """
    with open(params_path, "r") as f:
        model_params = json.load(f)
    
    config = LlamaConfig(**model_params)

    # Use GPT-2 tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    print("✅ Using GPT-2 tokenizer for testing.")
    
    # Load model with config
    model = LlamaForCausalLM(config)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model, tokenizer

# 2. Define Two-Layer Transformer Decoder
class TwoLayerTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        self.output_layer = nn.Linear(d_model, 50257)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        output = self.decoder(embedded, memory=torch.zeros_like(embedded))
        return self.output_layer(output)

# 3. Create and Split Random Dataset
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze()
        }

# Generate synthetic dataset
def create_synthetic_dataset(num_samples=5000):
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Deep learning models require vast amounts of data.",
        "Knowledge distillation improves model efficiency.",
        "Transformers are widely used in NLP applications."
    ]
    return [random.choice(sample_texts) for _ in range(num_samples)]

# 4. Define Knowledge Distillation Loss
class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        #soft_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=-1), F.softmax(teacher_logits / self.temperature, dim=-1))
        #soft_loss=self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=-1),F.softmax(teacher_logits[:, :student_logits.shape[1], :] / self.temperature, dim=-1))
        soft_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=-1),F.softmax(teacher_logits[:, :student_logits.shape[1], :student_logits.shape[2]] / self.temperature, dim=-1))
        hard_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1), ignore_index=-100)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

# 5. Train the Student Model
def train_student_model(student_model, teacher_model, train_loader, optimizer, loss_fn, device):
    student_model.to(device)
    teacher_model.to(device)
    for epoch in range(10):
        student_model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids, attention_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits
            #student_logits = student_model(input_ids=input_ids, attention_mask=attention_mask).logits
            student_logits = student_model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(student_logits, torch.Tensor):
                student_logits = student_logits.squeeze(0)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100
            loss = loss_fn(student_logits, teacher_logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# 6. Main Execution
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = r"C:\Users\Kaushiki\Downloads\project\Knowledge_distillation\llama3.2\Llama3.2-1B\consolidated.00.pth"
    tokenizer_path = r"path/to/tokenizer.model"
    params_path = r"C:\Users\Kaushiki\Downloads\project\Knowledge_distillation\llama3.2\Llama3.2-1B\params.json"
    
    
    teacher_model, tokenizer = load_teacher_model(model_path, tokenizer_path, params_path, device)
    student_model = TwoLayerTransformerDecoder(vocab_size=len(tokenizer))
    
    texts = create_synthetic_dataset()
    train_size = int(0.8 * len(texts))
    val_size = int(0.1 * len(texts))
    test_size = len(texts) - train_size - val_size
    
    train_texts, val_texts, test_texts = random_split(texts, [train_size, val_size, test_size])
    train_dataset = TextDataset(train_texts, tokenizer)
    val_dataset = TextDataset(val_texts, tokenizer)
    test_dataset = TextDataset(test_texts, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    optimizer = optim.AdamW(student_model.parameters(), lr=5e-5)
    loss_fn = DistillationLoss()
    train_student_model(student_model, teacher_model, train_loader, optimizer, loss_fn, device)

if __name__ == "__main__":
    main()
