import os

class Config:
    # Paths (align with main repo)
    DATA_PATH = '/content/drive/MyDrive/LUNA16/'  # Adjust for local/Colab
    ANNOTATIONS_FILE = os.path.join(DATA_PATH, 'annotations.csv')
    CANDIDATES_FILE = os.path.join(DATA_PATH, 'candidates.csv')
    OUTPUT_PATH = os.path.join(DATA_PATH, 'eda_outputs/')
    SAMPLE_SCAN_PATH = os.path.join(OUTPUT_PATH, 'sample_analysis/')  # For saved figures
    
    # EDA Settings
    PATCH_SIZE = 64  # For patch extraction
    SEED = 42
    N_SAMPLES = 1000  # For balanced sampling preview
