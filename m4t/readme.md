# Irula-Malayalam Speech Translation with SeamlessM4T

This project implements fine-tuning of Meta's SeamlessM4T model for Malayalam-to-Irula speech translation. The project includes data preparation, model training, and evaluation scripts for both speech-to-text (S2T) and speech-to-speech (S2S) translation tasks.

## 🎯 Project Overview

- **Source Language**: Malayalam (mal)
- **Target Language**: Irula (iru)
- **Model**: SeamlessM4T v2 Large
- **Tasks**: Speech-to-Text (S2T) and Speech-to-Speech (S2S) translation
- **Data Format**: Paired audio files with text transcriptions

## 📁 Project Structure

```
irula/
├── ma/                     # Malayalam audio files (source)
├── la/                     # Irula audio files (target)
├── lama.csv               # Text data with transcriptions
├── manifests/             # Generated manifest files
│   ├── train_manifest.json
│   └── valid_manifest.json
├── checkpoints/           # Model checkpoints
└── scripts/               # Training and evaluation scripts
```

## 🛠️ Installation

### System Dependencies

```bash
# Update system packages
apt update

# Install audio processing libraries
apt install libsndfile1-dev ffmpeg sox libsox-dev
```

### Python Dependencies

```bash
# Install core dependencies
pip install fairseq2 pydub sentencepiece datasets=="2.18.0" whisper_normalizer

# Install audio processing libraries
pip install librosa soundfile

# Install evaluation metrics
pip install jiwer

# Install SeamlessM4T (clone from Meta's repository first)
pip install -e ./seamless_communication
```

## 📊 Data Preparation

### CSV Format

Your `lama.csv` should contain:
- `id of speech and text`: File identifier (without .wav extension)
- `Irula text`: Irula transcription
- `Malayalam text`: Malayalam transcription

### Audio Requirements

- **Sample Rate**: 16kHz (resampled if different)
- **Format**: WAV files
- **Channels**: Mono preferred
- **Naming**: Consistent between source and target folders

### Audio Resampling (Windows)

```cmd
# Resample audio files to 16kHz using FFmpeg
for /r "ma" %i in (*.wav) do ffmpeg -i "%i" -ar 16000 -ac 1 "%~dpni_16k.wav"
for /r "la" %i in (*.wav) do ffmpeg -i "%i" -ar 16000 -ac 1 "%~dpni_16k.wav"
```

## 🚀 Usage

### 1. Create Training Manifests

First, run the manifest creation script to prepare your data:

```python
python create_manifest_with_text.py
```

This will:
- Resample audio files to 16kHz
- Create train/validation splits (90/10)
- Generate manifest files in JSON format

### 2. Fine-tune the Model

#### Speech-to-Speech Translation

```bash
python seamless_communication/src/seamless_communication/cli/m4t/finetune/finetune.py \
    --train_dataset manifests/train_manifest.json \
    --eval_dataset manifests/valid_manifest.json \
    --model_name seamlessM4T_v2_large \
    --save_model_to ./finetuned_model \
    --batch_size 4 \
    --max_epochs 5 \
    --learning_rate 1e-5 \
    --mode SPEECH_TO_SPEECH \
    --device cuda
```

#### Speech-to-Text Translation

```bash
python seamless_communication/src/seamless_communication/cli/m4t/finetune/finetune.py \
    --train_dataset manifests/train_manifest.json \
    --eval_dataset manifests/valid_manifest.json \
    --model_name seamlessM4T_v2_large \
    --save_model_to ./finetuned_model \
    --batch_size 4 \
    --max_epochs 5 \
    --learning_rate 1e-5 \
    --mode SPEECH_TO_TEXT \
    --device cuda
```

#### Alternative Training Command

```bash
m4t_finetune \
  --train_dataset manifests/train_manifest.json \
  --eval_dataset manifests/valid_manifest.json \
  --batch_size 4 \
  --eval_steps 1000 \
  --learning_rate 0.00005 \
  --patience 10 \
  --save_model_to checkpoints/ft_gs_m4tM.pt \
  --model_name seamlessM4T_v2_large
```

### 3. Evaluate the Model

Run the evaluation script to compare base and fine-tuned models:

```python
python evaluate_irula_model.py
```

## 📈 Evaluation Metrics

The evaluation script provides:

- **WER (Word Error Rate)**: Percentage of words incorrectly predicted
- **CER (Character Error Rate)**: Percentage of characters incorrectly predicted
- **Comparison**: Base model vs. fine-tuned model performance
- **Task Coverage**: Both S2T and S2S evaluation

### Evaluation Configuration

```python
MAX_SAMPLES = 100  # Number of samples to evaluate
CHECKPOINT_PATH = "checkpoints/ft_gs_m4tM.pt"
EVAL_MANIFEST = "manifests/valid_manifest.json"
CSV_PATH = "lama.csv"
```

## 🔧 Training Parameters

### Recommended Settings

| Parameter | Value | Description |
|-----------|--------|-------------|
| `batch_size` | 2-4 | Reduce if encountering memory issues |
| `learning_rate` | 1e-5 | Conservative learning rate for fine-tuning |
| `max_epochs` | 5-10 | Adjust based on dataset size |
| `patience` | 10 | Early stopping patience |
| `eval_steps` | 1000 | Evaluation frequency |

### Memory Optimization

If encountering memory issues:

```bash
# Set environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use smaller batch size
--batch_size 1
```

## 🐛 Troubleshooting

### Common Issues

1. **Segmentation Fault**: Install `libsndfile1-dev` and reduce batch size
2. **Sample Rate Error**: Ensure all audio is 16kHz (use resampling script)
3. **CUDA Memory Error**: Reduce batch size or use CPU
4. **Missing Dependencies**: Install all required packages

### Error Solutions

```bash
# For audio loading issues
apt install libsndfile1-dev ffmpeg sox libsox-dev

# For CUDA memory issues
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# For package conflicts
pip install --upgrade torch torchaudio
```

## 📝 File Formats

### Manifest File Structure

```json
{
  "source": {
    "id": "file_id",
    "audio_local_path": "/path/to/malayalam/audio.wav",
    "text": "Malayalam transcription",
    "lang": "mal",
    "sampling_rate": 16000
  },
  "target": {
    "id": "file_id",
    "audio_local_path": "/path/to/irula/audio.wav",
    "text": "Irula transcription",
    "lang": "iru",
    "sampling_rate": 16000
  }
}
```

## 🎯 Expected Results

After fine-tuning, you should see:
- Improved WER and CER scores on Irula translation
- Better semantic understanding of Malayalam-Irula language pair
- Enhanced speech quality for S2S tasks

## 📚 References

- [SeamlessM4T Paper](https://arxiv.org/abs/2308.11596)
- [Meta SeamlessM4T Repository](https://github.com/facebookresearch/seamless_communication)
- [Fine-tuning Documentation](https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/finetune_README.md)

## 🤝 Contributing

1. Ensure data quality and consistency
2. Test with small datasets first
3. Monitor training metrics
4. Report issues with detailed logs

## 📄 License

This project follows the SeamlessM4T license terms. Please refer to Meta's repository for detailed licensing information.