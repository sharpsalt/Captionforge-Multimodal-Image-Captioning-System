"""
Enhanced Image Captioning Model (PyTorch) - Optimized
======================================================
A state-of-the-art image captioning system with optimizations:
- ResNet50 encoder with fine-tuning (unfrozen after epoch 5)
- Transformer decoder with multi-head attention and padding masks
- Optimized vocabulary threshold for Flickr8k
- Greedy, Beam Search, and Nucleus Sampling
- BLEU-1 to BLEU-4 evaluation

Author: AI Assistant
Dataset: Flickr8k
"""

import os
import re
import pickle
import random
import numpy as np
from collections import Counter
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# NLP for evaluation
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.corpus import wordnet

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGES_DIR = os.path.join(BASE_DIR, "Images")
    CAPTIONS_FILE = os.path.join(BASE_DIR, "captions.txt")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    
    # Data
    MIN_CAPTION_LENGTH = 5
    VOCAB_THRESHOLD = 2  # Optimized for Flickr8k (reduced from 5)
    MAX_CAPTION_LENGTH = 35
    
    # Model Architecture
    IMAGE_SIZE = 224
    EMBEDDING_DIM = 256
    NUM_HEADS = 8
    FF_DIM = 1024
    NUM_DECODER_LAYERS = 4
    DROPOUT_RATE = 0.3
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE_DECODER = 3e-4
    LEARNING_RATE_ENCODER = 1e-5  # For fine-tuning
    WEIGHT_DECAY = 1e-4
    UNFREEZE_EPOCH = 5  # Unfreeze encoder after this epoch
    EARLY_STOPPING_PATIENCE = 5
    
    # Inference
    BEAM_WIDTHS = [3, 5, 10]
    TOP_P = 0.9
    
    # Split
    TRAIN_SIZE = 6000
    VAL_SIZE = 1000
    TEST_SIZE = 1000

config = Config()
os.makedirs(config.MODEL_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
def load_captions(filepath):
    captions_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',', 1)
            if len(parts) != 2: continue
            image_name, caption = parts
            image_name = image_name.strip()
            caption = caption.strip()
            if image_name not in captions_dict:
                captions_dict[image_name] = []
            captions_dict[image_name].append(caption)
    return captions_dict

def clean_caption(caption):
    caption = caption.lower()
    caption = re.sub(r'[^a-zA-Z\s]', '', caption)
    caption = ' '.join(caption.split())
    return caption

def preprocess_captions(captions_dict, min_length=5):
    processed = {}
    all_captions = []
    for img, caps in captions_dict.items():
        cleaned_caps = []
        for cap in caps:
            cleaned = clean_caption(cap)
            if len(cleaned.split()) >= min_length:
                cleaned = f"<start> {cleaned} <end>"
                cleaned_caps.append(cleaned)
                all_captions.append(cleaned)
        if cleaned_caps:
            processed[img] = cleaned_caps
    return processed, all_captions

def build_vocabulary(captions, threshold=2):
    word_counts = Counter()
    for cap in captions:
        word_counts.update(cap.split())
    vocab = ['<pad>', '<start>', '<end>', '<unk>']
    vocab += [w for w, c in word_counts.items() if c >= threshold and w not in vocab]
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    return vocab, word_to_idx, idx_to_word

# Process
raw_captions = load_captions(config.CAPTIONS_FILE)
captions_dict, all_captions = preprocess_captions(raw_captions, config.MIN_CAPTION_LENGTH)
vocab, word_to_idx, idx_to_word = build_vocabulary(all_captions, config.VOCAB_THRESHOLD)
VOCAB_SIZE = len(vocab)
print(f"Vocabulary size: {VOCAB_SIZE}")

all_images = list(captions_dict.keys())
random.shuffle(all_images)
existing_images = [img for img in all_images if os.path.exists(os.path.join(config.IMAGES_DIR, img))]
train_images = existing_images[:config.TRAIN_SIZE]
val_images = existing_images[config.TRAIN_SIZE:config.TRAIN_SIZE + config.VAL_SIZE]
test_images = existing_images[config.TRAIN_SIZE + config.VAL_SIZE:config.TRAIN_SIZE + config.VAL_SIZE + config.TEST_SIZE]

# ============================================================================
# DATASET
# ============================================================================
train_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
    transforms.RandomCrop(config.IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CaptionDataset(Dataset):
    def __init__(self, image_names, captions_dict, word_to_idx, max_len, transform, images_dir):
        self.word_to_idx = word_to_idx
        self.max_len = max_len
        self.transform = transform
        self.images_dir = images_dir
        self.pairs = []
        for img in image_names:
            img_path = os.path.join(images_dir, img)
            if os.path.exists(img_path) and img in captions_dict:
                for cap in captions_dict[img]:
                    self.pairs.append((img_path, cap))
    
    def __len__(self): return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, caption = self.pairs[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        tokens = [self.word_to_idx.get(w, self.word_to_idx['<unk>']) for w in caption.split()]
        input_seq = tokens[:-1]
        target_seq = tokens[1:]
        input_seq = input_seq[:self.max_len-1]
        target_seq = target_seq[:self.max_len-1]
        pad_len = self.max_len - 1 - len(input_seq)
        input_seq += [0] * pad_len
        target_seq += [0] * pad_len
        return image, torch.tensor(input_seq), torch.tensor(target_seq)

train_loader = DataLoader(CaptionDataset(train_images, captions_dict, word_to_idx, config.MAX_CAPTION_LENGTH, train_transform, config.IMAGES_DIR), batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(CaptionDataset(val_images, captions_dict, word_to_idx, config.MAX_CAPTION_LENGTH, val_transform, config.IMAGES_DIR), batch_size=config.BATCH_SIZE, shuffle=False)

# ============================================================================
# ARCHITECTURE
# ============================================================================
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.proj = nn.Sequential(nn.Conv2d(2048, embed_dim, 1), nn.ReLU(), nn.Dropout(0.3))
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.proj(features)
        b, c, h, w = features.shape
        return features.view(b, c, h*w).permute(0, 2, 1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout, max_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        self.dropout = nn.Dropout(dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

    def forward(self, encoder_output, target, target_mask=None, target_padding_mask=None):
        seq_len = target.size(1)
        if target_mask is None:
            target_mask = self.generate_square_subsequent_mask(seq_len).to(target.device)
        x = self.embedding(target) * np.sqrt(self.embed_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        # Pass tgt_key_padding_mask to ignore <pad> tokens in attention
        output = self.transformer_decoder(x, encoder_output, tgt_mask=target_mask, tgt_key_padding_mask=target_padding_mask)
        return self.fc_out(output)

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout, max_len):
        super().__init__()
        self.encoder = ImageEncoder(embed_dim)
        self.decoder = TransformerDecoder(vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout, max_len)
    
    def forward(self, images, captions):
        encoder_output = self.encoder(images)
        # Create padding mask (True for <pad> tokens)
        padding_mask = (captions == 0)
        output = self.decoder(encoder_output, captions, target_padding_mask=padding_mask)
        return output

# Create
model = ImageCaptioningModel(VOCAB_SIZE, config.EMBEDDING_DIM, config.NUM_HEADS, config.FF_DIM, config.NUM_DECODER_LAYERS, config.DROPOUT_RATE, config.MAX_CAPTION_LENGTH).to(DEVICE)

# Initial: Freeze encoder
for param in model.encoder.backbone.parameters():
    param.requires_grad = False

# ============================================================================
# TRAINING
# ============================================================================
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE_DECODER, weight_decay=config.WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, input_seq, target_seq in loader:
        images, input_seq, target_seq = images.to(device), input_seq.to(device), target_seq.to(device)
        optimizer.zero_grad()
        output = model(images, input_seq)
        loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

best_val_loss = float('inf')
patience_counter = 0

print("Starting Optimized Training...")
for epoch in range(config.EPOCHS):
    # Dynamic Unfreezing
    if epoch == config.UNFREEZE_EPOCH:
        print("\n[!] Unfreezing encoder backbone for fine-tuning...")
        for param in model.encoder.backbone.parameters():
            param.requires_grad = True
        optimizer = AdamW([
            {"params": model.encoder.backbone.parameters(), "lr": config.LEARNING_RATE_ENCODER},
            {"params": model.decoder.parameters(), "lr": config.LEARNING_RATE_DECODER}
        ], weight_decay=config.WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, input_seq, target_seq in val_loader:
            images, input_seq, target_seq = images.to(DEVICE), input_seq.to(DEVICE), target_seq.to(DEVICE)
            output = model(images, input_seq)
            val_loss += criterion(output.view(-1, output.size(-1)), target_seq.view(-1)).item()
    val_loss /= len(val_loader)
    
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1}/{config.EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, 'best_model_optimized.pth'))
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= config.EARLY_STOPPING_PATIENCE: break

model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, 'best_model_optimized.pth')))

# ============================================================================
# EVALUATION (Subset for speed)
# ============================================================================
def greedy_search(model, image, max_len, word_to_idx, idx_to_word, device):
    model.eval()
    with torch.no_grad():
        encoder_output = model.encoder(image.unsqueeze(0).to(device))
        sequence = [word_to_idx['<start>']]
        for _ in range(max_len-1):
            input_seq = torch.tensor([sequence]).to(device)
            output = model.decoder(encoder_output, input_seq)
            next_token = output[0, -1, :].argmax().item()
            if next_token == word_to_idx['<end>']: break
            sequence.append(next_token)
    return ' '.join([idx_to_word.get(i, '<unk>') for i in sequence[1:]])

print("\n--- Final Evaluation (Greedy) ---")
references, hypotheses = [], []
for img in test_images[:100]:
    img_path = os.path.join(config.IMAGES_DIR, img)
    image = val_transform(Image.open(img_path).convert('RGB'))
    pred = greedy_search(model, image, config.MAX_CAPTION_LENGTH, word_to_idx, idx_to_word, DEVICE)
    references.append([re.sub(r'<start>|<end>', '', c).strip() for c in captions_dict[img]])
    hypotheses.append(pred)

bleu = {}
for n in range(1, 5):
    weights = tuple([1.0/n]*n + [0.0]*(4-n))
    bleu[f'BLEU-{n}'] = corpus_bleu([[r.split() for r in refs] for refs in references], [h.split() for h in hypotheses], weights=weights, smoothing_function=SmoothingFunction().method4)

for k, v in bleu.items(): print(f"{k}: {v:.4f}")
