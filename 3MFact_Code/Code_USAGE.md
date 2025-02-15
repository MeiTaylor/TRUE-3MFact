# 3MFact Code Usage Guide

## Overview
3MFact is a framework for multimodal fact verification utilizing Multi-role Multimodal Models. This guide provides instructions for using the 3MFact framework with the TRUE dataset for video fact-checking tasks.

## Prerequisites
- Python 3.9.18 or higher
- CUDA-enabled GPU (recommended)
- Access to the TRUE dataset
- OpenAI API key
- Google API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MeiTaylor/TRUE-3MFact.git
cd TRUE-3MFact/3MFact_Code
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
- MiniCPM-V-2.6 (for both VideoLMM and ImageLMM)
- en_core_web_sm-3.7.1

## Configuration

1. Update `Config.py` with your settings:
```python
MODEL_CONFIG = {
    "llm": {
        "model_name": "your-model",
        "model_type": "OpenAI",
        "model_key": "your-api-key"
    }
    # ... other model configurations
}
```

2. Configure dataset paths in `DATASET_CONFIG`:
```python
DATASET_CONFIG = {
    "root_dir": "/path/to/TRUE_Dataset",
    # ... other dataset paths
}
```

## Usage

### Running the Pipeline

Execute the main pipeline:
```bash
python main_Pipeline.py
```


## Evaluation

Run the evaluation script to assess model performance:
```bash
python Evaluation.py
```

Results are logged in `Evaluation.log` and include:
- Accuracy metrics
- Average evaluation scores
- Detailed performance statistics

## Directory Structure
```
3MFact_Framework/
├── Evaluation_Relate/     # Evaluation related codes
├── Katna/                 # Video frame extraction
├── ClaimVerifier.py      # Claim verification module
├── Config.py             # Configuration settings
├── Evaluation.py         # Evaluation script
├── InformationRetriever.py # Information retrieval
├── Pipeline.py           # Main pipeline
└── ...
```


## License
The codes are licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
