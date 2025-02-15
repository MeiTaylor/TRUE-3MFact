# config.py

# Model Configurations
MODEL_CONFIG = {
    "llm": {
        "model_name": "gpt-4o-mini",
        "model_type": "OpenAI",
        "model_key": "sk-P5XNXJl13d7dc49f30D7T3BLBKFJ7cb0ca6A5908472aBCBf"
    },
    "video_lmm": {
        "model_name": "MiniCPM-V-2.6",
        "model_type": "MiniCPM",
        "local_model_path": "/Data/niukaipeng/pretrained_model/OpenBMB/MiniCPM-V-2_6"
    },
    "image_lmm": {
        "model_name": "MiniCPM-V-2.6",
        "model_type": "MiniCPM",
        "local_model_path": "/Data/niukaipeng/pretrained_model/OpenBMB/MiniCPM-V-2_6"
    },
    "en_core_web_sm":{
        "model_name": "en_core_web_sm-3.7.1",
        "model_type": "Spacy",
        "local_model_path": "/home/public/FakeNews/nkp/3MFact_Framework_213/model/en_core_web_sm-3.7.1"
    }
}

# Video Descriptor Configurations
VIDEO_DESCRIPTOR_CONFIG = {
    "keyframes_per_video": 7,  # Number of keyframes to extract per video
    "keyframe_extractor": "katna"  # Method used for keyframe extraction
}

# Information Retriever Configurations
INFORMATION_RETRIEVER_CONFIG = {
    "max_links_per_item": 10,  # Maximum number of evidence links per search item
    "max_search_items": 2,     # Default number of key search items
    "max_evidences": 3,        # Top evidences to select
    "relevance_window": 250,   # Tokens window for relevance checking
    "scoring_weights": {
        "website_quality": 0.25,   # Weight for website quality score
        "newness": 0.25,          # Weight for newness score
        "relevance": 0.50         # Weight for relevance score
    },
    "max_iterations": 3
}

# Claim Verifier Configurations
CLAIM_VERIFIER_CONFIG = {
    "confidence_threshold": 0.93,  # Threshold for confidence level
    "max_iterations": 3           # Maximum verification iterations
}

# Question Manager Configurations
QUESTION_MANAGER_CONFIG = {
    "max_iterations": 3,  # Maximum number of question generation iterations
}

# Pipeline Configurations
PIPELINE_CONFIG = {
    "max_cycles": 3,  # Maximum number of cycles for the entire pipeline
}

# Path Configurations
PATH_CONFIG = {
    "data_dir": "test_data/",
    "cache_dir": "cache/",
}



# API Related Configurations
API_CONFIG = {
    "google_api_key": "sk-P5XNXJl13d7dc49f30D7T3BLBKFJ7cb0ca6A5908472aBCBf",
}





# Dataset Configurations
DATASET_CONFIG = {
    # Root directory path (relative to Config.py)
    "root_dir": "/home/public/FakeNews/nkp/TRUE_Dataset",
   
    # Annotation file paths (relative to root_dir)
    "annotation": {
        "test": "test_set.txt",
        "train_val": "train_val_set.txt"
    },
   
    # Data directory paths (relative to root_dir)
    "data_dir": {
        "test": "test",
        "train_val": "train_val"
    },
   
    # Video directory paths (relative to root_dir)
    "video_dir": {
        "test": "test_video",
        "train_val": "train_val_video"
    },

    # Output directory paths (relative to root_dir)
    "output_dir": {
        "test": "test_output",
        "train_val": "train_val_output"
    },

    "evaluation_output_dir":{
        "test": "evaluation_test_output",
        "train_val": "evaluation_train_val_output"
    }

}