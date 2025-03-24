import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM, LlamaConfig
import numpy as np
from tqdm import tqdm
import argparse
import os
import time
from typing import List, Dict, Tuple, Optional, Union

# --------------------------------
# Load Configuration and Model
# --------------------------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_llama_model(model_dir, tokenizer_dir, config_path, device='cuda'):
    """
    Load the Llama3.2 teacher model and tokenizer.
    """
    # Load model configuration
    config = LlamaConfig.from_json_file(config_path)
    print("Configuration:", config)

    # Load tokenizer using PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    print("Tokenizer:", tokenizer)

    # Set the padding token to eos_token
    tokenizer.pad_token = tokenizer.eos_token
    # Load model weights
    model = LlamaForCausalLM.from_pretrained(model_dir, config=config, trust_remote_code=True)
    model.to(device)
    model.eval()

    return model, tokenizer

def load_teacher_model(config):
    model_dir = "/local/mnt/workspace/kdwivedi/model"
    tokenizer_dir = "/local/mnt/workspace/kdwivedi/model/tokenizer"
    config_path = "/local/mnt/workspace/kdwivedi/model/config.json"
    model, tokenizer = load_llama_model(model_dir, tokenizer_dir, config_path, device=config["device"])
    return model, tokenizer

# --------------------------------
# Dataset class
# --------------------------------
# class TextDataset(Dataset):
#     def __init__(self, tokenizer, data_path, max_length):
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#         # Load and tokenize the data
#         with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
#             text = f.read()

#         # Split text into chunks of max_length for training
#         self.examples = []
#         for i in range(0, len(text), max_length):
#             chunk = text[i:i + max_length * 2]  # Overlap to ensure we have enough context
#             if len(chunk) > max_length:
#                 self.examples.append(chunk)

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, idx):
#         chunk = self.examples[idx]
#         encodings = self.tokenizer(chunk, truncation=True, max_length=self.max_length,
#                                    padding="max_length", return_tensors="pt")

#         input_ids = encodings["input_ids"].squeeze()
#         attention_mask = encodings["attention_mask"].squeeze()

#         # For causal language modeling, labels are the same as inputs
#         labels = input_ids.clone()

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": labels
#         }

class TextDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load and parse the JSON data
        with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
            self.examples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        instruction = example["instruction"]
        context = example["context"]
        response = example["response"]

        # Combine instruction, context, and response for tokenization
        text = f"Instruction: {instruction} Context: {context} Response: {response}"
        encodings = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                   padding="max_length", return_tensors="pt")

        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()

        # For causal language modeling, labels are the same as inputs
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# --------------------------------
# Single-layer Transformer Decoder (Student Model)
# --------------------------------
class SingleLayerTransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])

        # Position embeddings
        self.position_embeddings = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config["hidden_size"],
            num_heads=config["num_attention_heads"],
            batch_first=True
        )

        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(config["hidden_size"], config["hidden_size"] * 4),
            nn.GELU(),
            nn.Linear(config["hidden_size"] * 4, config["hidden_size"]),
            nn.Dropout(0.1)
        )

        # Layer norms
        self.layer_norm1 = nn.LayerNorm(config["hidden_size"])
        self.layer_norm2 = nn.LayerNorm(config["hidden_size"])

        # Output layer
        self.output_layer = nn.Linear(config["hidden_size"], config["vocab_size"])

        # Dropout
        self.dropout = nn.Dropout(0.1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()

        # Create position ids
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)

        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Create causal mask
        causal_mask = self.generate_square_subsequent_mask(seq_length).to(input_ids.device)

        # Handle padding mask
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (1 - attention_mask).bool()

        # Self-attention
        attn_output, _ = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )

        # Residual connection and layer norm
        hidden_states = hidden_states + attn_output
        hidden_states = self.layer_norm2(hidden_states)

        # Feed-forward network
        ff_output = self.ff_network(hidden_states)

        # Residual connection
        hidden_states = hidden_states + ff_output

        # Get logits
        logits = self.output_layer(hidden_states)

        return logits

# --------------------------------
# Knowledge Distillation Loss
# --------------------------------
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    # KL divergence loss
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl_div = F.kl_div(log_probs, soft_targets, reduction='batchmean') * (temperature ** 2)

    # Task-specific loss (standard cross-entropy)
    ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1), ignore_index=-100)

    # Combined loss
    loss = alpha * kl_div + (1 - alpha) * ce_loss
    return loss

# --------------------------------
# Text Generation Functions
# --------------------------------
def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, config=None):
    model.eval()

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config["device"])
    attention_mask = torch.ones_like(input_ids)

    # Generate text auto-regressively
    generated = input_ids.clone()

    for _ in range(max_length):
        # Get predictions
        with torch.no_grad():
            if isinstance(model, SingleLayerTransformerDecoder):
                outputs = model(generated, attention_mask)
            else:
                outputs = model(generated, attention_mask=attention_mask)
                if not isinstance(outputs, torch.Tensor):
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                    elif isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]

            next_token_logits = outputs[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text

def evaluate_model(model, tokenizer, test_data_path, config):
    model.eval()
    try:
        # Prepare test dataset
        test_dataset = TextDataset(tokenizer, test_data_path, config["max_length"])
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        print(f"Falling back to training data: {config['data_path']}")
        test_dataset = TextDataset(tokenizer, config["data_path"], config["max_length"])
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
        print(f"Test data path: {config['test_data_path']}")

    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(config["device"])
            attention_mask = batch["attention_mask"].to(config["device"])
            labels = batch["labels"].to(config["device"])
            try:
                # Forward pass
                if isinstance(model, SingleLayerTransformerDecoder):
                    outputs = model(input_ids, attention_mask)
                else:
                    model_outputs = model(input_ids, attention_mask=attention_mask)
                    outputs = model_outputs.logits
                # Calculate loss
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1), ignore_index=-100)
                total_loss += loss.item()
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue
    if len(test_loader) == 0:
        print("No valid batches for evaluation")
        return 0, 0

    avg_loss = total_loss / len(test_loader)
    perplexity = np.exp(avg_loss)

    print(f"Evaluation results: Loss = {avg_loss:.4f}, Perplexity = {perplexity:.4f}")
    return avg_loss, perplexity

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compare_models(student_model, teacher_model, tokenizer, prompts, config):
    """Compare text generation between student and teacher models"""
    print("\nComparing text generation between student and teacher models:")

    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")

        # Time and generate with student model
        start_time = time.time()
        student_text = generate_text(student_model, tokenizer, prompt, max_length=50, temperature=1, config=config)
        student_time = time.time() - start_time

        # Time and generate with teacher model
        start_time = time.time()
        teacher_text = generate_text(teacher_model, tokenizer, prompt, max_length=50, temperature=1, config=config)
        teacher_time = time.time() - start_time

        # Print results
        print(f"Student ({student_time:.2f}s): {student_text}")
        print(f"Teacher ({teacher_time:.2f}s): {teacher_text}")
        print(f"Generation speedup: {teacher_time / student_time:.2f}x")

    # Count parameters
    student_params = count_parameters(student_model)
    teacher_params = count_parameters(teacher_model)
    print(f"\nModel size comparison:")
    print(f"Student parameters: {student_params:,}")
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Compression ratio: {teacher_params / student_params:.2f}x")

# --------------------------------
# Training Functions
# --------------------------------
# def train_with_knowledge_distillation(config):
#     """Main training function for knowledge distillation"""
#     # Load the teacher model
#     teacher_model, tokenizer = load_teacher_model(config)
#     print(f"Teacher model loaded: {config['teacher_model_name']}")

#     # Initialize the student model
#     student_model = SingleLayerTransformerDecoder(config).to(config["device"])
#     print("Student model initialized")

#     # Prepare dataset
#     train_dataset = TextDataset(tokenizer, config["data_path"], config["max_length"])
#     train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
#     print(f"Dataset loaded with {len(train_dataset)} examples")

#     # Optimizer
#     optimizer = optim.AdamW(student_model.parameters(), lr=config["learning_rate"])

#     # Training loop
#     print("Starting training...")
#     for epoch in range(config["epochs"]):
#         student_model.train()
#         total_loss = 0

#         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
#         for batch in progress_bar:
#             # Move batch to device
#             input_ids = batch["input_ids"].to(config["device"])
#             attention_mask = batch["attention_mask"].to(config["device"])
#             labels = batch["labels"].to(config["device"])

#             # Zero gradients
#             optimizer.zero_grad()

#             # Forward pass through student model
#             student_logits = student_model(input_ids, attention_mask)

#             # Get teacher's predictions
#             with torch.no_grad():
#                 teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask)
#                 teacher_logits = teacher_outputs.logits

#             # Calculate distillation loss
#             loss = distillation_loss(
#                 student_logits,
#                 teacher_logits,
#                 labels,
#                 config["temperature"],
#                 config["alpha"]
#             )

#             # Backpropagation
#             loss.backward()
#             optimizer.step()

#             # Update progress bar
#             total_loss += loss.item()
#             progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

#         # Save the model after each epoch
#         torch.save(student_model.state_dict(), config["save_path"])
#         print(f"Epoch {epoch+1} completed. Model saved to {config['save_path']}")

#     print("Training completed!")
#     return student_model

def train_with_knowledge_distillation(config):
    """Main training function for knowledge distillation"""
    # Load the teacher model
    teacher_model, tokenizer = load_teacher_model(config)
    print(f"Teacher model loaded: {config['teacher_model_name']}")

    # Initialize the student model
    student_model = SingleLayerTransformerDecoder(config).to(config["device"])
    print("Student model initialized")

    # Prepare dataset
    train_dataset = TextDataset(tokenizer, config["data_path"], config["max_length"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    print(f"Dataset loaded with {len(train_dataset)} examples")

    # Optimizer
    optimizer = optim.AdamW(student_model.parameters(), lr=config["learning_rate"])

    # Training loop
    print("Starting training...")
    for epoch in range(config["epochs"]):
        student_model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(config["device"])
            attention_mask = batch["attention_mask"].to(config["device"])
            labels = batch["labels"].to(config["device"])

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass through student model
            student_logits = student_model(input_ids, attention_mask)

            # Get teacher's predictions
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits

            # Calculate distillation loss
            loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                config["temperature"],
                config["alpha"]
            )

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

        # Save the model after each epoch
        torch.save(student_model.state_dict(), config["save_path"])
        print(f"Epoch {epoch+1} completed. Model saved to {config['save_path']}")

    print("Training completed!")
    return student_model
# --------------------------------
# Main Function
# --------------------------------

def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation from pretrained LLM to Single-Layer Transformer")

    parser.add_argument("--mode", choices=["train", "evaluate", "generate"], default="train",
                        help="Mode: train, evaluate, or generate text")
    parser.add_argument("--data", default="synthetic_data.txt",
                        help="Path to training data file")
    parser.add_argument("--test_data", default=None,
                        help="Path to test data file (if not specified, uses training data)")
    parser.add_argument("--teacher", default="llama3.2",
                        help="Teacher model name")
    parser.add_argument("--model_path", default="student_model.pt",
                        help="Path to saved student model (for evaluation or generation)")
    parser.add_argument("--save_path", default="student_model.pt",
                        help="Path to save the student model (for training)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--max_len", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--temp", type=float, default=2.0,
                        help="Temperature for distillation")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for KL divergence loss")
    parser.add_argument("--prompt", default="Once upon a time",
                        help="Prompt for text generation")
    parser.add_argument("--gen_max_len", type=int, default=100,
                        help="Maximum length for generated text")
    parser.add_argument("--config_path", default="config.json",
                        help="Path to the configuration file")

    args = parser.parse_args()

    if args.test_data is None:
        args.test_data = args.data

    # Load config from config.json
    config = load_config(args.config_path)

    # Add teacher_model_name to config
    config["teacher_model_name"] = args.teacher

    # Update config with command line arguments
    config["data_path"] = args.data
    config["test_data_path"] = args.test_data
    config["save_path"] = args.save_path
    config["batch_size"] = args.batch_size
    config["epochs"] = args.epochs
    config["learning_rate"] = args.lr
    config["temperature"] = args.temp
    config["alpha"] = args.alpha
    config["max_length"] = args.max_len
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize device
    print(f"Using device: {config['device']}")

    try:
        if args.mode == "train":
            print("Starting knowledge distillation training...")
            student_model = train_with_knowledge_distillation(config)

            # Optionally evaluate after training
            teacher_model, tokenizer = load_teacher_model(config)
            print("\nEvaluating trained model:")
            evaluate_model(student_model, tokenizer, config["test_data_path"], config)

        elif args.mode == "evaluate":
            print(f"Evaluating model from {args.model_path}...")

            # Load teacher and tokenizer
            teacher_model, tokenizer = load_teacher_model(config)

            # Load student model
            student_model = SingleLayerTransformerDecoder(config)
            student_model.load_state_dict(torch.load(args.model_path, map_location=config["device"]))
            student_model.to(config["device"])

            # Evaluate
            student_loss, student_ppl = evaluate_model(student_model, tokenizer, config["test_data_path"], config)
            teacher_loss, teacher_ppl = evaluate_model(teacher_model, tokenizer, config["test_data_path"], config)

            print(f"\nPerplexity comparison:")
            print(f"Student model perplexity: {student_ppl:.2f}")
            print(f"Teacher model perplexity: {teacher_ppl:.2f}")
            print(f"Relative performance: {(student_ppl / teacher_ppl):.2f}x worse than teacher")

            # Compare text generation
            prompts = [
                "The Forbidden forest was dark and discovered",
                "Harry raised his wand and shouted",
                "Hermione opened the ancient book and shouted",
            ]

            compare_models(student_model, teacher_model, tokenizer, prompts, config)

        elif args.mode == "generate":
            print(f"Generating text using model from {args.model_path}...")

            # Load teacher and tokenizer
            teacher_model, tokenizer = load_teacher_model(config)

            # Load student model
            student_model = SingleLayerTransformerDecoder(config)
            student_model.load_state_dict(torch.load(args.model_path, map_location=config["device"]))
            student_model.to(config["device"])

            # Generate text
            print(f"Prompt: {args.prompt}")
            student_text = generate_text(student_model, tokenizer, args.prompt, max_length=args.gen_max_len, config=config)
            print(f"Student generated: {student_text}")

            teacher_text = generate_text(teacher_model, tokenizer, args.prompt, max_length=args.gen_max_len, config=config)
            print(f"Teacher generated: {teacher_text}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation from pretrained LLM to Single-Layer Transformer")

    parser.add_argument("--mode", choices=["train", "evaluate", "generate"], default="train",
                        help="Mode: train, evaluate, or generate text")
    parser.add_argument("--data", default="synthetic_data.txt",
                        help="Path to training data file")
    parser.add_argument("--test_data", default=None,
                        help="Path to test data file (if not specified, uses training data)")
    parser.add_argument("--teacher", default="llama3.2",
                        help="Teacher model name")
    parser.add_argument("--model_path", default="student_model.pt",
                        help="Path to saved student model (for evaluation or generation)")
    parser.add_argument("--save_path", default="student_model.pt",
                        help="Path to save the student model (for training)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--max_len", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--temp", type=float, default=2.0,
                        help="Temperature for distillation")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for KL divergence loss")
    parser.add_argument("--prompt", default="Once upon a time",
                        help="Prompt for text generation")
    parser.add_argument("--gen_max_len", type=int, default=100,
                        help="Maximum length for generated text")
    parser.add_argument("--config_path", default="config.json",
                        help="Path to the configuration file")

    args = parser.parse_args()

    if args.test_data is None:
        args.test_data = args.data

    # Load config from config.json
    config = load_config(args.config_path)

    # Update config with command line arguments
    config["data_path"] = args.data
    config["test_data_path"] = args.test_data
    config["save_path"] = args.save_path
    config["batch_size"] = args.batch_size
    config["epochs"] = args.epochs
    config["learning_rate"] = args.lr
    config["temperature"] = args.temp
    config["alpha"] = args.alpha
    config["max_length"] = args.max_len
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize device
    print(f"Using device: {config['device']}")

    try:
        if args.mode == "train":
            print("Starting knowledge distillation training...")
            student_model = train_with_knowledge_distillation(config)

            # Optionally evaluate after training
            teacher_model, tokenizer = load_teacher_model(config)
            print("\nEvaluating trained model:")
            evaluate_model(student_model, tokenizer, config["test_data_path"], config)

        elif args.mode == "evaluate":
            print(f"Evaluating model from {args.model_path}...")

            # Load teacher and tokenizer
            teacher_model, tokenizer = load_teacher_model(config)

            # Load student model
            student_model = SingleLayerTransformerDecoder(config)
            student_model.load_state_dict(torch.load(args.model_path, map_location=config["device"]))
            student_model.to(config["device"])

            # Evaluate
            student_loss, student_ppl = evaluate_model(student_model, tokenizer, config["test_data_path"], config)
            teacher_loss, teacher_ppl = evaluate_model(teacher_model, tokenizer, config["test_data_path"], config)

            print(f"\nPerplexity comparison:")
            print(f"Student model perplexity: {student_ppl:.2f}")
            print(f"Teacher model perplexity: {teacher_ppl:.2f}")
            print(f"Relative performance: {(student_ppl / teacher_ppl):.2f}x worse than teacher")

            # Compare text generation
            prompts = [
                "The Forbidden forest was dark and discovered",
                "Harry raised his wand and shouted",
                "Hermione opened the ancient book and shouted",
            ]

            compare_models(student_model, teacher_model, tokenizer, prompts, config)

        elif args.mode == "generate":
            print(f"Generating text using model from {args.model_path}...")

            # Load teacher and tokenizer
            teacher_model, tokenizer = load_teacher_model(config)

            # Load student model
            student_model = SingleLayerTransformerDecoder(config)
            student_model.load_state_dict(torch.load(args.model_path, map_location=config["device"]))
            student_model.to(config["device"])

            # Generate text
            print(f"Prompt: {args.prompt}")
            student_text = generate_text(student_model, tokenizer, args.prompt, max_length=args.gen_max_len, config=config)
            print(f"Student generated: {student_text}")

            teacher_text = generate_text(teacher_model, tokenizer, args.prompt, max_length=args.gen_max_len, config=config)
            print(f"Teacher generated: {teacher_text}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''/local/mnt/workspace/kdwivedi/model$
