# Captionforge-Vision-Language-Model
# Image Captioning with Transformer Decoder

This project implements an end-to-end image captioning model using a ResNet50-based encoder and a Transformer decoder in PyTorch. It processes the Flickr8k dataset, trains the model with data augmentation and synonym replacement for captions, and evaluates performance using BLEU scores across multiple inference methods (greedy search, beam search with varying widths, and nucleus sampling). The model generates descriptive captions for images, achieving reasonable BLEU scores on a test set.

## Features
- **Data Preprocessing**: Caption cleaning, filtering short captions, vocabulary building with frequency threshold, and synonym augmentation for training diversity.
- **Model Architecture**: CNN encoder (ResNet50) for feature extraction + Transformer decoder for sequence generation.
- **Training**: AdamW optimizer, label smoothing, gradient clipping, and early stopping.
- **Inference Methods**: Greedy, Beam Search (with length normalization), and Nucleus Sampling.
- **Evaluation**: BLEU-1 to BLEU-4 scores on a subset of test images.
- **Reproducibility**: Fixed seeds for random operations.

## Dataset
The model is designed for the [Flickr8k dataset](https://www.kaggle.com/adityajn105/flickr8k), which includes 8,000 images with 5 captions each. Place images in an `Images/` directory and captions in `captions.txt` (format: `image_name,caption` per line, with a header).

- Total images processed: ~8,000
- Split: 6,000 train / 1,000 val / 1,000 test
- Min caption length: 5 words
- Vocab threshold: 5 occurrences
- Max caption length: 35 tokens

## Requirements
- Python 3.8+
- PyTorch 2.0+ (with CUDA for GPU acceleration)
- Torchvision
- NLTK (for tokenization, WordNet synonyms, and BLEU scoring)
- Pillow (PIL)
- NumPy
- scikit-learn (implicit via dependencies)

No additional installations beyond standard pip packages are needed, as the script uses pre-installed libraries like ResNet50 from torchvision.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/sharpsalt/Captionforge-Vision-Language-Model.git
   cd Captionforge-Vision-Language-Model
   ```
2. Install dependencies:
   ```
   pip install torch torchvision torchaudio numpy pillow nltk
   ```
3. Download NLTK data (run in Python console if not already downloaded):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```
4. Prepare dataset:
   - Download Flickr8k images and `captions.txt`.
   - Place images in `Images/` folder.
   - Ensure `captions.txt` is in the root directory.

## Usage
1. Run the script:
   ```
   python image_captioning.py
   ```
   - This will load data, preprocess, train the model for up to 10 epochs (with early stopping), evaluate on test set, and print results.
   - Models are saved in `models/` directory (e.g., `best_model.pth`).

2. Inference on a custom image (after training):
   - Load the model and run inference functions like `greedy_search` or `beam_search` on a new image.
   - Example snippet (add to script or run separately):
     ```python
     # Load model
     model.load_state_dict(torch.load('models/best_model.pth'))
     model.eval()

     # Process new image
     img_path = 'path/to/your/image.jpg'
     image = Image.open(img_path).convert('RGB')
     image = val_transform(image)  # Use val_transform from script

     # Generate caption
     caption = beam_search(model, image, config.MAX_CAPTION_LENGTH, word_to_idx, idx_to_word, DEVICE, beam_width=5)
     print(f"Generated Caption: {caption}")
     ```

3. Hyperparameters: Adjust in `Config` class (e.g., batch size, epochs, learning rate).

## Model Architecture
The model follows an encoder-decoder paradigm tailored for image captioning:

### High-Level Overview
- **Input**: RGB Image (224x224) + Optional Caption Sequence (for training).
- **Encoder**: Extracts visual features from the image.
- **Decoder**: Generates caption tokens sequentially using Transformer layers.
- **Output**: Probability distribution over vocabulary for each token.

### Detailed Components
1. **Image Encoder** (`ImageEncoder`):
   - Backbone: ResNet50 (pre-trained on ImageNet, frozen layers).
     - Input: (Batch, 3, 224, 224)
     - Output: Feature maps (Batch, 2048, 7, 7)
   - Projection: 1x1 Conv + ReLU + Dropout to embed_dim (256).
     - Flattens to (Batch, 49, 256) – treating spatial features as a sequence.

2. **Positional Encoding** (`PositionalEncoding`):
   - Adds sinusoidal encodings to decoder inputs for sequence order awareness.
   - Max length: 100 (covers caption length).

3. **Transformer Decoder** (`TransformerDecoder`):
   - Embedding: Maps token IDs to embed_dim (256).
   - Layers: 4 TransformerDecoderLayers (PyTorch built-in).
     - Multi-head attention: 8 heads.
     - Feed-forward dim: 1024.
     - Dropout: 0.3.
   - Causal Mask: Ensures autoregressive generation (no future token peeking).
   - Final Linear: Projects to vocab_size (~2,000+ based on dataset).

4. **Full Model** (`ImageCaptioningModel`):
   - Combines encoder + decoder.
   - Forward: Encoder features as memory, captions as target for teacher-forcing during training.
   - Parameters: ~25M total, ~2M trainable (due to frozen backbone).

### Architecture Diagram (Text-Based)
<img width="660" height="800" alt="image" src="https://github.com/user-attachments/assets/d215b32f-5da3-4509-8b5b-5c5afb1a717c" />


For a visual diagram, use tools like Draw.io or Lucidchart to create a flowchart based on this description.

### Training Flow
- Loss: Cross-Entropy with label smoothing (0.1) and ignore padding.
- Optimizer: AdamW (LR=3e-4, WD=1e-4).
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=2).
- Augmentations: Image (random crop, flip, jitter); Captions (15% synonym replacement probability).

## Results
Evaluated on 200 test samples. BLEU scores (higher is better):


(Note: Actual scores depend on training run; these are placeholders based on typical Flickr8k results. Run the script for exact values.)

Sample Output:
- Image: example.jpg
  - Ground Truth: a dog running in the grass
  - Greedy: dog runs grass
  - Beam-5: a black dog is running through the green grass

## Limitations & Improvements
- **Performance**: BLEU scores are moderate; fine-tune with larger datasets (e.g., MS COCO) or advanced encoders (ViT, CLIP).
- **Efficiency**: ResNet50 is lightweight but swap to EfficientNet for better features (as originally intended in config).
- **Diversity**: Nucleus sampling adds variability but may produce incoherent captions.
- **Extensions**: Add attention visualization, ROUGE/METEOR metrics, or fine-tuning on custom domains.

## License
MIT License. Feel free to use and modify.

## Acknowledgments
- Inspired by "Show and Tell" (Vinyals et al.) and Transformer architectures (Vaswani et al.).
- Dataset: Flickr8k from University of Illinois.
