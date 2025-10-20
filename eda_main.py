import os
import logging
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import cv2
import SimpleITK as sitk
from tqdm import tqdm
import plotly.express as px  # Interactive plots
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas  # PDF report

from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seeds
np.random.seed(Config.SEED)

def load_metadata():
    """Load annotations and candidates CSVs."""
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    annotations = pd.read_csv(Config.ANNOTATIONS_FILE)
    candidates = pd.read_csv(Config.CANDIDATES_FILE)
    logger.info(f"Annotations shape: {annotations.shape}, Candidates shape: {candidates.shape}")
    return annotations, candidates

def visualize_class_distribution(candidates):
    """Plot class distribution."""
    class_dist = candidates['class'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=class_dist.index, y=class_dist.values, ax=ax)
    ax.set_title('Candidate Class Distribution')
    ax.set_xlabel('Class (0: Non-Nodule, 1: Nodule)')
    ax.set_ylabel('Count')
    plt.savefig(os.path.join(Config.OUTPUT_PATH, 'class_distribution.png'), dpi=150)
    plt.close()
    logger.info("Class distribution plot saved.")

def load_sample_scan(seriesuid):
    """Load and normalize a sample CT scan."""
    mhd_files = glob.glob(os.path.join(Config.DATA_PATH, 'subset*/', f'{seriesuid}.mhd'))
    if not mhd_files:
        raise ValueError(f"No .mhd file for {seriesuid}")
    itkimage = sitk.ReadImage(mhd_files[0])
    ct_scan = sitk.GetArrayFromImage(itkimage).astype(np.float32) - 1024  # Hounsfield
    logger.info(f"Sample scan shape: {ct_scan.shape}, Range: [{ct_scan.min():.1f}, {ct_scan.max():.1f}] HU")
    return ct_scan, itkimage

def visualize_sample_slices(ct_scan, n_slices=10):
    """Visualize axial slices."""
    mid_slice = ct_scan.shape[0] // 2
    start, end = max(0, mid_slice - n_slices//2), min(ct_scan.shape[0], mid_slice + n_slices//2 + 1)
    fig, axes = plt.subplots(1, n_slices, figsize=(20, 4))
    for i, z in enumerate(range(start, end)):
        axes[i].imshow(ct_scan[z], cmap='gray', vmin=-1000, vmax=400)
        axes[i].set_title(f'Slice {z}')
        axes[i].axis('off')
    plt.suptitle('Sample CT Axial Slices')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_PATH, 'sample_slices.png'), dpi=150)
    plt.close()
    logger.info("Sample slices plot saved.")

def world_to_voxel_coords(coords, origin, spacing):
    """Convert world to voxel coordinates."""
    return np.round((coords - origin) / spacing).astype(int)

def overlay_nodules(ct_scan, annotations, seriesuid, origin, spacing):
    """Overlay nodule markers on middle slice."""
    sample_nodules = annotations[annotations['seriesuid'] == seriesuid]
    mid_slice = ct_scan.shape[0] // 2
    slice_img = ct_scan[mid_slice]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(slice_img, cmap='gray', vmin=-1000, vmax=400)
    for _, nodule in sample_nodules.iterrows():
        coords = np.array([nodule['coordX'], nodule['coordY'], nodule['coordZ']])
        voxel = world_to_voxel_coords(coords, origin, spacing)
        if abs(voxel[0] - mid_slice) < 5:
            ax.plot(voxel[1], voxel[2], 'r+', markersize=10, markeredgewidth=2)
    ax.set_title('Middle Slice with Nodule Markers')
    ax.axis('off')
    plt.savefig(os.path.join(Config.OUTPUT_PATH, 'nodule_overlay.png'), dpi=150)
    plt.close()
    logger.info("Nodule overlay plot saved.")

def extract_patch(ct_scan, coords, origin, spacing, patch_size=Config.PATCH_SIZE):
    """Extract and resize patch."""
    voxel = world_to_voxel_coords(coords, origin, spacing)
    z, y, x = np.clip(voxel, [0, 0, 0], [ct_scan.shape[0]-1, ct_scan.shape[1]-1, ct_scan.shape[2]-1])
    half = patch_size // 2
    y_start, y_end = np.clip([y - half, y + half], 0, ct_scan.shape[1])
    x_start, x_end = np.clip([x - half, x + half], 0, ct_scan.shape[2])
    patch = ct_scan[z, y_start:y_end, x_start:x_end]
    if patch.shape != (patch_size, patch_size):
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    return patch

def visualize_patches(annotations, candidates, ct_scan, origin, spacing):
    """Visualize sample nodule and non-nodule patches."""
    # Nodule patch
    if len(annotations) > 0:
        nodule_row = annotations.iloc[0]
        nodule_coords = np.array([nodule_row['coordX'], nodule_row['coordY'], nodule_row['coordZ']])
        nodule_patch = extract_patch(ct_scan, nodule_coords, origin, spacing)
    
    # Non-nodule patch
    non_nodule_candidates = candidates[candidates['class'] == 0]
    if len(non_nodule_candidates) > 0:
        non_nodule_row = non_nodule_candidates.iloc[0]
        non_nodule_coords = np.array([non_nodule_row['coordX'], non_nodule_row['coordY'], non_nodule_row['coordZ']])
        non_nodule_patch = extract_patch(ct_scan, non_nodule_coords, origin, spacing)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if 'nodule_patch' in locals():
        axes[0].imshow(nodule_patch, cmap='gray', vmin=-1000, vmax=400)
        axes[0].set_title('Nodule Patch')
        axes[0].axis('off')
    if 'non_nodule_patch' in locals():
        axes[1].imshow(non_nodule_patch, cmap='gray', vmin=-1000, vmax=400)
        axes[1].set_title('Non-Nodule Patch')
        axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_PATH, 'sample_patches.png'), dpi=150)
    plt.close()
    logger.info("Sample patches plot saved.")

def dataset_statistics(annotations, candidates):
    """Compute and plot dataset stats."""
    # Scans per subset
    subsets = glob.glob(os.path.join(Config.DATA_PATH, 'subset*'))
    scan_counts = {os.path.basename(s): len(glob.glob(os.path.join(s, '*.mhd'))) for s in subsets}
    logger.info(f"Scans per subset: {scan_counts}")
    
    # Nodules per scan
    nodules_per_scan = annotations['seriesuid'].value_counts()
    logger.info(f"Nodules per scan - Min: {nodules_per_scan.min()}, Max: {nodules_per_scan.max()}, Mean: {nodules_per_scan.mean():.1f}")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    nodules_per_scan.hist(bins=20, ax=ax)
    ax.set_title('Distribution of Nodules per Scan')
    ax.set_xlabel('Number of Nodules')
    ax.set_ylabel('Frequency')
    plt.savefig(os.path.join(Config.OUTPUT_PATH, 'nodules_distribution.png'), dpi=150)
    plt.close()
    logger.info("Dataset stats plot saved.")

def balanced_sampling_preview(annotations, candidates, n_samples=Config.N_SAMPLES):
    """Preview balanced sampling and splits."""
    positive_count = min(len(annotations), n_samples)
    negative_candidates = candidates[candidates['class'] == 0].sample(n=positive_count, random_state=Config.SEED)
    balanced_negatives = negative_candidates[['seriesuid', 'coordX', 'coordY', 'coordZ', 'class']]
    positive_samples = annotations[['seriesuid', 'coordX', 'coordY', 'coordZ']].copy()
    positive_samples['class'] = 1
    all_samples = pd.concat([positive_samples, balanced_negatives], ignore_index=True)
    
    train_samples, test_samples = train_test_split(all_samples, test_size=0.2, random_state=Config.SEED, stratify=all_samples['class'])
    
    logger.info(f"Balanced samples: {len(all_samples)} (Positives: {positive_count})")
    logger.info(f"Train: {len(train_samples)} ({np.sum(train_samples['class'] == 1)} positives)")
    logger.info(f"Test: {len(test_samples)} ({np.sum(test_samples['class'] == 1)} positives)")
    
    # Interactive split plot
    split_df = pd.DataFrame({'Split': ['Train', 'Test'] * 2, 'Class': ['Positive']*2 + ['Negative']*2,
                             'Count': [np.sum(train_samples['class'] == 1), np.sum(test_samples['class'] == 1),
                                       len(train_samples) - np.sum(train_samples['class'] == 1),
                                       len(test_samples) - np.sum(test_samples['class'] == 1)]})
    fig = px.bar(split_df, x='Split', y='Count', color='Class', title='Balanced Sampling Split Preview')
    fig.write_html(os.path.join(Config.OUTPUT_PATH, 'sampling_preview.html'))
    logger.info("Sampling preview HTML saved.")

def generate_pdf_report():
    """Generate a summary PDF report."""
    c = canvas.Canvas(os.path.join(Config.OUTPUT_PATH, 'eda_report.pdf'), pagesize=letter)
    c.drawString(100, 750, "LUNA16 EDA Report")
    c.drawString(100, 730, "Key Insights: Highly imbalanced classes; sparse nodules; HU range -1000 to 400.")
    # Add more text or embed images if needed
    c.save()
    logger.info("PDF report generated.")

def run_eda():
    """Full EDA pipeline."""
    annotations, candidates = load_metadata()
    visualize_class_distribution(candidates)
    
    # Sample analysis
    sample_seriesuid = annotations['seriesuid'].iloc[0]
    ct_scan, itkimage = load_sample_scan(sample_seriesuid)
    visualize_sample_slices(ct_scan)
    origin = np.array(itkimage.GetOrigin())
    spacing = np.array(itkimage.GetSpacing())
    overlay_nodules(ct_scan, annotations, sample_seriesuid, origin, spacing)
    visualize_patches(annotations, candidates, ct_scan, origin, spacing)
    
    dataset_statistics(annotations, candidates)
    balanced_sampling_preview(annotations, candidates)
    generate_pdf_report()
    
    logger.info(f"EDA complete. Outputs in {Config.OUTPUT_PATH}")

if __name__ == "__main__":
    # Mount Drive in Colab if needed
    run_eda()
