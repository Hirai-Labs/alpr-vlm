# Indonesian ALPR with Visual Language Model


## Dataset Overview

This sections contains the dataset downloader component for the Automatic License Plate Recognition (ALPR) system using Vision Language Models (VLM). The dataset is specifically structured for fine-tuning VLM models for ALPR tasks.

The dataset is structured for instruction-based learning with the following features:
- Instructions in natural language
- Vehicle metadata including:
  - Color
  - License plate number
  - Make (manufacturer)
  - Model
  - Vehicle type
- Associated vehicle images

### Dataset Statistics
- Total examples: 330
- Dataset size: ~56.1 MB
- Download size: ~55.6 MB

### Dataset Structure

The dataset follows this structure:
```yaml
features:
  - instruction: string
  - output:
      - color: string
      - license_plate: string
      - make: string
      - model: string
      - type: string
  - image: sequence[image]

splits:
  train:
    num_examples: 330
    num_bytes: 56056906.0
```

### Getting Started

#### Prerequisites
```bash
pip install -r requirements.txt
```

#### Downloading the Dataset

The dataset files are organized in the following structure:
```
data/
  ├── train-00000-of-00001.parquet
  └── ...
```

To download the dataset:
```bash
python download_dataset.py
```

### Usage

After downloading, you can load the dataset using:

```python
from datasets import load_dataset

dataset = load_dataset("Hirai-Labs/alpr-vlm-instruct-dataset")
```

Each example in the dataset contains:
- An instruction prompt
- Vehicle metadata (color, license plate, make, model, type)
- Associated vehicle image

### Dataset Format

Example data point:---
dataset_info:
  features:
  - name: instruction
    dtype: string
  - name: output
    struct:
    - name: color
      dtype: string
    - name: license_plate
      dtype: string
    - name: make
      dtype: string
    - name: model
      dtype: string
    - name: type
      dtype: string
  - name: image
    sequence: image
  splits:
  - name: train
    num_bytes: 56056906.0
    num_examples: 330
  download_size: 55646391
  dataset_size: 56056906.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
```python
{
    'instruction': 'Analyze this vehicle image and provide its details',
    'output': {
        'color': 'White',
        'license_plate': 'B 1234 CD',
        'make': 'Toyota',
        'model': 'Camry',
        'type': 'Car'
    },
    'image': [Image]
}
```