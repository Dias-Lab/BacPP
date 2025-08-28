# BPP: BacterialPloidyPredictor

BacterialPloidyPredictor is the first package that aims to predict the bacterial polyploidy utilizing the complete genome assemblies. It allows users to apply one of the three machine learning models we trained for binary prediction. 

## Installation

To maximize the use of this tool, installation of CheckM2 is recommended. Although this BPP is fully functional without CheckM2, the incorporation of CheckM2 gives another metrics for the prediction confidence. According to our result, genome assemblies below 88% completeness would result in significant shift in feature values extracted for subsequent prediction. To enable the incorporation of CheckM2, add --checkm2. (See below for details)

### Prerequisites

Ensure you have following packages:

 - Python 3.9+
 - Numpy
 - Pandas
 - Matplotlib
 - sklearn
 - xgboost
 - statsmodels

### Option 1: Through GitHub repository

```bash
git clone https://github.com/Dias-Lab/BacterialPloidyPredictor-v1.git
conda env create -n bpp -f bpp.yml
conda activate bpp
```

### Option 2: through pip install

```bash
pip install 
```



---

## Overview

<img src="paper/figures/BacterialPloidyPredictor-flowchart.png">

- **Global genome features extraction** - 1.1 to 2.10

- **Genome completeness estimation through CheckM2** - 1.3

- **Binary monoploidy/polyploidy prediction** - 2.11

- **Confidence estimation through multiple visualizations and quantitative metrics** - 2.12 to 3.6

## Usage

If installed via `pip`, use bpp `...`; if installed via `git clone`, use `bpp.py ...`

To pull out the instruction page for full list of arguments, 
```bash
bpp -h
```
```bash
usage: bpp [-h] [--predict] [--out OUT] [--cpus CPUS] [--images] [--checkm2] [--model {knn,lg,xgb}] [--model-path MODEL_PATH] [--num-windows NUM_WINDOWS]
           [--no-interactions] [--id-col ID_COL] [--pred-input PRED_INPUT] [--pred-output PRED_OUTPUT]
           folder

BPP: Bactererial Ploidy Predictor to identify bacteiral polyploidy based on global genomic architecture.

positional arguments:
  folder                Input folder containing FASTA/FA/FNA files

options:
  -h, --help            show this help message and exit
  --predict             After feature extraction, run polyploidy prediction using a trained model.
  --out OUT             Output directory (default: <folder>/outputs)
  --cpus CPUS           Number of CPU cores (1=serial).
  --images              Generate GC/AT skew images into <output folder>/image
  --checkm2             Run CheckM2 and append completeness/contamination to predictions.csv.
  --model {knn,lg,xgb}  Model to use for prediction if --predict is set. Default: knn
  --model-path MODEL_PATH
                        Path to model file (defaults to ./models/kNNPC.json / ./models/MLG.json / ./models/XGBoost.json).
  --num-windows NUM_WINDOWS
                        Number of windows for extracting global genomic architecture (default: 4096)
  --no-interactions     Do not add interaction terms
  --id-col ID_COL       ID column name in the features CSV for prediction. Default: file
  --pred-input PRED_INPUT
                        Optional: features CSV to use for prediction (overrides --out).
  --pred-output PRED_OUTPUT
                        Optional: predictions CSV path (2 columns: ID, polyploidy_pred). Default: <features_csv_dir>/predictions.csv
```
## Note



## Outputs
<img src="paper/figures/example_outputs.png">
Above is the illustration from a selection of outputs from examples using genomes from *E.coli*, *Synechocystis sp. PCC 6803*, and *Citribacter freundii* Chromosome of *C.freundii* is visualized in forms of global GC skew and AT skew in circular representation (a) and linear representation (b). In panel (a), the outer ring represents global GC skew with positive and negative skew values in orange and purple colors, respecively; the inner ring represents global AT skew with positive and negative skew values in olive and grey colors, respectively. In panel (b), cumulative GC skew (blue) and AT skew (orange) are illustrated in different color. Additionally, two more plots are provided to help users assess the confidence of the prediction. (c) an interactive 3-dimentional PCA plot is generated and saved as the .html file. Below is a screenshot of the illustration, where orange (group 1: monoploidy), green (group 2: polyploidy with similar genomic architecture with group 1), and blue (polyploidy with divergent architecture in contrast to group 1) represents three groups of reference bacterial species with known ploidy. Samples for prediction is included in the scatter plot as black points. The name of the sample will be appear upon hovering above the point. (d) Pairwise Euclidean distance for reference samples within the same group (black) and between two different groups (blue) are enumerated which forms two distributions. The Euclidean distance between every sample for prediction and its nearest reference sample is calculated and included on the histogram to help assess the confidence of the prediction. 

| file | polyploidy_pred | PED.confidence | completeness | contamination |
| :-------: | :------: | :-------: | :-------: |  :-------: |  
| Citrobacter_freundii_ATCC_8090.fasta | 1 | '0.9767195925928703 | | |
| Escherichia_coli.fna | 0.725 | 0 | 1 | | |
| Synechocystis_sp_PCC_6803.fasta | 1 | 1 | | |

The main prediction and prediction confidence are saved in the main output file prediction.csv. Above is the example output when kNN (default) is selected as the model for prediction:

