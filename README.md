
# LUNA16 Exploratory Data Analysis

**Description**: Comprehensive EDA for the LUNA16 dataset, including metadata inspection, CT scan visualization, nodule overlays, patch extraction, and balanced sampling previews. Designed to inform preprocessing and model development.

## Setup
1. Clone: `git clone https://github.com/yourusername/luna16-eda.git`
2. Install: `pip install -r requirements.txt`
3. Configure paths in `config.py`.
4. Run: `python eda_main.py`

## Outputs
- Plots: Class distributions, slices, patches, nodule histograms.
- Interactive: Sampling preview (HTML).
- Report: PDF summary.

## Integration
- Use with main repo for end-to-end pipeline.
- Key Finding: Severe class imbalance (recommend oversampling).

## License
MIT
