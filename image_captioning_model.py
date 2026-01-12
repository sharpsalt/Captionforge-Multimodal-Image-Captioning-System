import os
import re
import pickle
import random
import numpy as np
from collections import Counter
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Importing pyTorch Utilities Function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Importing NLP Utilities Function
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
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print(f"Using device: {DEVICE}") #gpu, but still for checking 

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
    MIN_CAPTION_LENGTH = 5 #Since it is Flickr8k,i kept it 5, but ideally for the more accuracy it should be 3
    VOCAB_THRESHOLD = 5 #Same situation for this too
    MAX_CAPTION_LENGTH = 35
    
    # Model Architecture
    IMAGE_SIZE = 224  # EfficientNet input
    EMBEDDING_DIM = 256
    NUM_HEADS = 8
    FF_DIM = 1024
    NUM_DECODER_LAYERS = 4
    DROPOUT_RATE = 0.3
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 10 #It is not a hyperparameter, it is just for the ease of training
    LEARNING_RATE = 3e-4 
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 5
    
    # Inference
    BEAM_WIDTHS = [3, 5, 10]
    TOP_P = 0.9
    
    # Split xuz we already know that it contains 8k Images and Captions as well
    TRAIN_SIZE = 6000
    VAL_SIZE = 1000
    TEST_SIZE = 1000

config = Config()
os.makedirs(config.MODEL_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
print("="*60)
print("PHASE 1: DATA LOADING AND PREPROCESSING")
print("="*60)

"""The whole logic behind it is that in captions.txt file,the first Line is Header,
and then from the Second it is Image Name and Caption which is seperated by , so i splitted it on the basis of it
and then checking whether is avoided 1st Line or not if yes then i am loading it into captions_dict"""
def load_captions(filepath):
    """Load and parse captions file"""
    captions_dict = {}
    with open(filepath,'r',encoding='utf-8') as f:
        next(f) # Skip header
        for line in f:
            line=line.strip()
            if not line:
                continue
            parts=line.split(',', 1)
            if len(parts)!=2:
                continue
            image_name,caption=parts
            image_name=image_name.strip()
            caption=caption.strip()
            if image_name not in captions_dict:
                captions_dict[image_name]=[]
            captions_dict[image_name].append(caption)
    return captions_dict

"""Making it compatible for any NLP Task first we have to clean the caption like removing the special characters and making it lower case
and then removing the extra spaces, and joining it by only 1 spaces because NLP treats each of it as a token"""
def clean_caption(caption):
    """Clean and preprocess a caption"""
    caption=caption.lower()
    caption=re.sub(r'[^a-zA-Z\s]','',caption)
    caption=' '.join(caption.split())
    return caption

"""What i am doing here is that i am checking whether the word is in the dictionary or not if yes then i am replacing it with its synonym for better performance
because we know how does it impactful word is but our System don't know that's why it is necessary"""
def augment_caption_synonyms(caption, prob=0.15):
    """Augment caption by replacing words with synonyms"""
    words=caption.split()
    augmented_words=[]
    for word in words:
        if random.random() < prob and word not in ['<start>','<end>']:
            synonyms=[]
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name().lower()!=word and '_' not in lemma.name():
                        synonyms.append(lemma.name().lower())
            if synonyms:
                word = random.choice(synonyms[:5])
            # print(synonyms)
        augmented_words.append(word)
    return ' '.join(augmented_words)

"""What i am doing here is that i am checking whether the word is in the dictionary or not if yes then i am replacing it with its synonym for better performance
because we know how does it impactful word is but our System don't know that's why it is necessary"""
def preprocess_captions(captions_dict, min_length=5):
    """Clean captions and filter short ones"""
    processed={}
    all_captions=[]
    
    for img,caps in captions_dict.items():
        cleaned_caps=[]
        for cap in caps:
            cleaned=clean_caption(cap)
            words=cleaned.split()
            if len(words)>=min_length:
                cleaned=f"<start> {cleaned} <end>"
                cleaned_caps.append(cleaned)
                all_captions.append(cleaned)
        
        if cleaned_caps:
            processed[img] = cleaned_caps
    
    return processed, all_captions

def build_vocabulary(captions, threshold=5):
    """Build vocabulary with frequency threshold"""
    word_counts=Counter()
    for cap in captions:
        words=cap.split()
        word_counts.update(words)
    
    vocab=['<pad>', '<start>', '<end>', '<unk>']
    vocab+=[w for w, c in word_counts.items() if c>=threshold and w not in vocab]
    
    word_to_idx={w: i for i, w in enumerate(vocab)}
    idx_to_word={i: w for w, i in word_to_idx.items()}
    return vocab, word_to_idx, idx_to_word

# Load and process captions
print("Loading captions...")
raw_captions = load_captions(config.CAPTIONS_FILE)
print(f"Loaded {len(raw_captions)} images with captions")

print("Preprocessing captions...")
captions_dict,all_captions=preprocess_captions(
    raw_captions,min_length=config.MIN_CAPTION_LENGTH
)
print(f"After filtering: {len(captions_dict)} images, {len(all_captions)} captions")

print("Building vocabulary...")
vocab,word_to_idx,idx_to_word=build_vocabulary(
    all_captions,threshold=config.VOCAB_THRESHOLD
)
VOCAB_SIZE=len(vocab)
print(f"Vocabulary size: {VOCAB_SIZE}")

# Find max caption length
max_length=max(len(cap.split()) for cap in all_captions)
config.MAX_CAPTION_LENGTH=min(max_length, config.MAX_CAPTION_LENGTH)
print(f"Max caption length: {config.MAX_CAPTION_LENGTH}")

# ============================================================================
# DATA SPLITTING
# ============================================================================
print("\nSplitting data...")
all_images=list(captions_dict.keys())
random.shuffle(all_images)
existing_images=[img for img in all_images if os.path.exists(os.path.join(config.IMAGES_DIR, img))]
print(f"Found {len(existing_images)} existing images")
train_images=existing_images[:config.TRAIN_SIZE]
val_images=existing_images[config.TRAIN_SIZE:config.TRAIN_SIZE + config.VAL_SIZE]
test_images=existing_images[config.TRAIN_SIZE + config.VAL_SIZE:
                               config.TRAIN_SIZE + config.VAL_SIZE + config.TEST_SIZE]

print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

# ============================================================================
# IMAGE TRANSFORMS
# ============================================================================
train_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
    transforms.RandomCrop(config.IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

val_transform=transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# DATASET
# ============================================================================
class CaptionDataset(Dataset):
    """Dataset for image captioning"""
    def __init__(self, image_names, captions_dict, word_to_idx, max_len, transform, images_dir, augment=False):
        self.word_to_idx=word_to_idx
        self.max_len=max_len
        self.transform=transform
        self.images_dir=images_dir
        self.augment=augment
        
        # Create (image_path, caption) pairs
        self.pairs=[]
        for img in image_names:
            img_path=os.path.join(images_dir, img)
            if os.path.exists(img_path) and img in captions_dict:
                for cap in captions_dict[img]:
                    self.pairs.append((img_path,cap))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, caption=self.pairs[idx]
        
        # Load image
        image=Image.open(img_path).convert('RGB')
        image=self.transform(image)
        
        # Augment caption
        if self.augment and random.random()<0.3:
            caption=augment_caption_synonyms(caption)
        
        # Tokenize
        tokens=[self.word_to_idx.get(w, self.word_to_idx['<unk>']) 
                  for w in caption.split()]
        
        # Input/target
        input_seq=tokens[:-1]
        target_seq=tokens[1:]
        
        # Pad
        input_seq=input_seq[:self.max_len - 1]
        target_seq=target_seq[:self.max_len - 1]
        
        pad_len=self.max_len - 1 - len(input_seq)
        input_seq=input_seq + [0] * pad_len
        target_seq=target_seq + [0] * pad_len
        
        return image, torch.tensor(input_seq), torch.tensor(target_seq)

# Create datasets
print("\nCreating datasets...")
train_dataset=CaptionDataset(
    train_images,captions_dict,word_to_idx,
    config.MAX_CAPTION_LENGTH,train_transform,config.IMAGES_DIR,augment=True
)
val_dataset=CaptionDataset(
    val_images,captions_dict,word_to_idx,
    config.MAX_CAPTION_LENGTH,val_transform,config.IMAGES_DIR,augment=False
)
test_dataset=CaptionDataset(
    test_images,captions_dict,word_to_idx,
    config.MAX_CAPTION_LENGTH,val_transform,config.IMAGES_DIR,augment=False
)

train_loader=DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
val_loader=DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
print("\n"+"="*60)
print("PHASE 2: BUILDING MODEL")
print("="*60)

class ImageEncoder(nn.Module):
    """EfficientNet-based image encoder"""
    def __init__(self, embed_dim):
        super().__init__()
        # Use ResNet50 (more compatible than EfficientNet)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]  # Remove FC and avgpool
        self.backbone = nn.Sequential(*modules)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Project to embedding dimension
        self.proj = nn.Sequential(
            nn.Conv2d(2048, embed_dim, 1),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        # x: (batch, 3, H, W)
        features = self.backbone(x)  # (batch, 2048, 7, 7)
        features = self.proj(features)  # (batch, embed_dim, 7, 7)
        batch, c, h, w = features.shape
        features = features.view(batch, c, h * w).permute(0, 2, 1)  # (batch, 49, embed_dim)
        return features

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerDecoder(nn.Module):
    """Transformer decoder for caption generation"""
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout, max_len):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        self.dropout = nn.Dropout(dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def forward(self, encoder_output, target, target_mask=None):
        # target: (batch, seq_len)
        seq_len = target.size(1)
        
        # Create causal mask
        if target_mask is None:
            target_mask = self.generate_square_subsequent_mask(seq_len).to(target.device)
        
        # Embed and add positional encoding
        x = self.embedding(target) * np.sqrt(self.embed_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Decode
        output = self.transformer_decoder(x, encoder_output, tgt_mask=target_mask)
        output = self.fc_out(output)
        
        return output

class ImageCaptioningModel(nn.Module):
    """Complete image captioning model"""
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout, max_len):
        super().__init__()
        self.encoder = ImageEncoder(embed_dim)
        self.decoder = TransformerDecoder(vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout, max_len)
    
    def forward(self, images, captions):
        encoder_output = self.encoder(images)
        output = self.decoder(encoder_output, captions)
        return output

# Create model
print("Building model...")
model = ImageCaptioningModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=config.EMBEDDING_DIM,
    num_heads=config.NUM_HEADS,
    ff_dim=config.FF_DIM,
    num_layers=config.NUM_DECODER_LAYERS,
    dropout=config.DROPOUT_RATE,
    max_len=config.MAX_CAPTION_LENGTH
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# TRAINING
# ============================================================================
print("\n" + "=" * 60)
print("PHASE 3: TRAINING")
print("=" * 60)

criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, input_seq, target_seq) in enumerate(loader):
        images = images.to(device)
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        
        optimizer.zero_grad()
        output = model(images, input_seq)
        
        # Reshape for loss: (batch * seq_len, vocab_size)
        output = output.view(-1, output.size(-1))
        target_seq = target_seq.view(-1)
        
        loss = criterion(output, target_seq)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, input_seq, target_seq in loader:
            images = images.to(device)
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            output = model(images, input_seq)
            output = output.view(-1, output.size(-1))
            target_seq = target_seq.view(-1)
            
            loss = criterion(output, target_seq)
            total_loss += loss.item()
    
    return total_loss / len(loader)

# Training loop
best_val_loss = float('inf')
patience_counter = 0

print("Starting training...")
for epoch in range(config.EPOCHS):
    print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
    
    train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss = validate(model, val_loader, criterion, DEVICE)
    
    scheduler.step(val_loss)
    
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, 'best_model.pth'))
        print("  -> Saved best model!")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# Load best model
model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, 'best_model.pth')))

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================
print("\n" + "=" * 60)
print("PHASE 4: INFERENCE")
print("=" * 60)

def greedy_search(model, image, max_len, word_to_idx, idx_to_word, device):
    """Generate caption using greedy search"""
    model.eval()
    
    start_token = word_to_idx['<start>']
    end_token = word_to_idx['<end>']
    
    with torch.no_grad():
        encoder_output = model.encoder(image.unsqueeze(0).to(device))
        
        sequence = [start_token]
        for _ in range(max_len - 1):
            input_seq = torch.tensor([sequence]).to(device)
            output = model.decoder(encoder_output, input_seq)
            next_token = output[0, -1, :].argmax().item()
            
            if next_token == end_token:
                break
            sequence.append(next_token)
    
    caption = ' '.join([idx_to_word.get(i, '<unk>') for i in sequence[1:]])
    return caption

def beam_search(model, image, max_len, word_to_idx, idx_to_word, device, beam_width=5):
    """Generate caption using beam search with length normalization"""
    model.eval()
    
    start_token = word_to_idx['<start>']
    end_token = word_to_idx['<end>']
    
    with torch.no_grad():
        encoder_output = model.encoder(image.unsqueeze(0).to(device))
        
        beams = [(0.0, [start_token])]
        completed = []
        
        for _ in range(max_len - 1):
            all_candidates = []
            
            for log_prob, sequence in beams:
                if sequence[-1] == end_token:
                    completed.append((log_prob, sequence))
                    continue
                
                input_seq = torch.tensor([sequence]).to(device)
                output = model.decoder(encoder_output, input_seq)
                log_probs = F.log_softmax(output[0, -1, :], dim=-1)
                
                top_k = torch.topk(log_probs, beam_width)
                for i in range(beam_width):
                    idx = top_k.indices[i].item()
                    new_log_prob = log_prob + top_k.values[i].item()
                    new_sequence = sequence + [idx]
                    all_candidates.append((new_log_prob, new_sequence))
            
            if not all_candidates:
                break
            
            # Length normalization
            all_candidates.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
            beams = all_candidates[:beam_width]
        
        completed.extend(beams)
        
        if completed:
            completed.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
            best_sequence = completed[0][1]
        else:
            best_sequence = [start_token]
    
    caption = ' '.join([idx_to_word.get(i, '<unk>') 
                        for i in best_sequence[1:] 
                        if i != end_token])
    return caption

def nucleus_sampling(model, image, max_len, word_to_idx, idx_to_word, device, top_p=0.9):
    """Generate caption using nucleus (top-p) sampling"""
    model.eval()
    
    start_token = word_to_idx['<start>']
    end_token = word_to_idx['<end>']
    
    with torch.no_grad():
        encoder_output = model.encoder(image.unsqueeze(0).to(device))
        
        sequence = [start_token]
        for _ in range(max_len - 1):
            input_seq = torch.tensor([sequence]).to(device)
            output = model.decoder(encoder_output, input_seq)
            probs = F.softmax(output[0, -1, :], dim=-1)
            
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=0)
            
            mask = cumsum_probs <= top_p
            mask[0] = True
            
            nucleus_probs = sorted_probs[mask]
            nucleus_indices = sorted_indices[mask]
            nucleus_probs = nucleus_probs / nucleus_probs.sum()
            
            idx = nucleus_indices[torch.multinomial(nucleus_probs, 1)].item()
            
            if idx == end_token:
                break
            sequence.append(idx)
    
    caption = ' '.join([idx_to_word.get(i, '<unk>') for i in sequence[1:]])
    return caption

# ============================================================================
# EVALUATION
# ============================================================================
print("\n" + "=" * 60)
print("PHASE 5: EVALUATION")
print("=" * 60)

def calculate_bleu_scores(references, hypotheses):
    """Calculate BLEU-1 to BLEU-4 scores"""
    smoothie = SmoothingFunction().method4
    
    ref_tokens = [[ref.split() for ref in refs] for refs in references]
    hyp_tokens = [hyp.split() for hyp in hypotheses]
    
    bleu_scores = {}
    for n in range(1, 5):
        weights = tuple([1.0/n] * n + [0.0] * (4-n))
        try:
            score = corpus_bleu(ref_tokens, hyp_tokens, weights=weights, 
                               smoothing_function=smoothie)
        except:
            score = 0.0
        bleu_scores[f'BLEU-{n}'] = score
    
    return bleu_scores

def evaluate_model(model, test_images, captions_dict, word_to_idx, idx_to_word, 
                   max_len, transform, images_dir, device, method='greedy', beam_width=5, num_samples=200):
    """Evaluate model on test set"""
    references = []
    hypotheses = []
    
    for img in test_images[:num_samples]:
        img_path = os.path.join(images_dir, img)
        if not os.path.exists(img_path) or img not in captions_dict:
            continue
        
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        
        true_captions = captions_dict[img]
        
        if method == 'greedy':
            pred_caption = greedy_search(model, image, max_len, word_to_idx, idx_to_word, device)
        elif method == 'beam':
            pred_caption = beam_search(model, image, max_len, word_to_idx, idx_to_word, device, beam_width)
        elif method == 'nucleus':
            pred_caption = nucleus_sampling(model, image, max_len, word_to_idx, idx_to_word, device)
        
        clean_refs = [re.sub(r'<start>|<end>', '', cap).strip() for cap in true_captions]
        references.append(clean_refs)
        hypotheses.append(pred_caption)
    
    bleu_scores = calculate_bleu_scores(references, hypotheses)
    return bleu_scores, references, hypotheses

# Evaluate
print("\nEvaluating model on test set...")
results = {}

# Greedy Search
print("\n--- Greedy Search ---")
bleu, refs, hyps = evaluate_model(
    model, test_images, captions_dict, word_to_idx, idx_to_word,
    config.MAX_CAPTION_LENGTH, val_transform, config.IMAGES_DIR, DEVICE, 
    method='greedy', num_samples=200
)
results['Greedy'] = bleu
for k, v in bleu.items():
    print(f"{k}: {v:.4f}")

# Beam Search
for k in config.BEAM_WIDTHS:
    print(f"\n--- Beam Search (k={k}) ---")
    bleu, refs, hyps = evaluate_model(
        model, test_images, captions_dict, word_to_idx, idx_to_word,
        config.MAX_CAPTION_LENGTH, val_transform, config.IMAGES_DIR, DEVICE,
        method='beam', beam_width=k, num_samples=200
    )
    results[f'Beam-{k}'] = bleu
    for key, v in bleu.items():
        print(f"{key}: {v:.4f}")

# Nucleus Sampling
print("\n--- Nucleus Sampling (p=0.9) ---")
bleu, refs, hyps = evaluate_model(
    model, test_images, captions_dict, word_to_idx, idx_to_word,
    config.MAX_CAPTION_LENGTH, val_transform, config.IMAGES_DIR, DEVICE,
    method='nucleus', num_samples=200
)
results['Nucleus'] = bleu
for k, v in bleu.items():
    print(f"{k}: {v:.4f}")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

print("\n{:<15} {:>10} {:>10} {:>10} {:>10}".format(
    "Method", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"
))
print("-" * 55)

for method, bleu in results.items():
    print("{:<15} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        method, bleu['BLEU-1'], bleu['BLEU-2'], bleu['BLEU-3'], bleu['BLEU-4']
    ))

# ============================================================================
# SAMPLE CAPTIONS
# ============================================================================
print("\n" + "=" * 60)
print("SAMPLE GENERATED CAPTIONS")
print("=" * 60)

for img in test_images[:5]:
    img_path = os.path.join(config.IMAGES_DIR, img)
    if not os.path.exists(img_path) or img not in captions_dict:
        continue
    
    print(f"\nImage: {img}")
    print("Ground Truth:")
    for cap in captions_dict[img][:2]:
        clean_cap = re.sub(r'<start>|<end>', '', cap).strip()
        print(f"  - {clean_cap}")
    
    image = Image.open(img_path).convert('RGB')
    image = val_transform(image)
    
    print("Generated:")
    print(f"  Greedy: {greedy_search(model, image, config.MAX_CAPTION_LENGTH, word_to_idx, idx_to_word, DEVICE)}")
    print(f"  Beam-5: {beam_search(model, image, config.MAX_CAPTION_LENGTH, word_to_idx, idx_to_word, DEVICE, 5)}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Model saved to: {os.path.join(config.MODEL_DIR, 'best_model.pth')}")
