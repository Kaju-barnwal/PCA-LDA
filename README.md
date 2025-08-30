🍷 Wine Quality Analysis with PCA & LDA

📌 Project Overview

This project applies Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) on the Wine Quality dataset (WineQT.csv) to explore dimensionality reduction and classification techniques in R.
Problem Statement: Wine quality is influenced by multiple physicochemical properties. Understanding how these features contribute to wine classification can help in quality control and production optimization.

Goal:

Reduce dimensionality using PCA and interpret the variance explained by principal components.
Classify wine quality using LDA (both multiclass and binary Fisher’s LDA).
Compare accuracy, visualize discriminant functions, and derive insights for real-world applications.

📊 Dataset Description – WineQT

Observations: 1,143 wines
Variables: 13 (12 features + 1 target variable + Id)

Features include:
Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, Free sulfur dioxide, total sulfur dioxide, density, pH, sulphates and alcohol

Target Variable: quality (integer score between 3–8, later treated as a factor for classification)
Id: Unique identifier (removed during modeling)

🛠️ Technologies & Libraries Used

R (base)
ggplot2 – Visualization (scatter plots, scree plots, projections)
MASS – LDA modeling
stats – PCA (prcomp), covariance & eigen decomposition

🔎 Methodology

1. Data Preprocessing
Removed ID column (not predictive).
Converted quality to a categorical factor for classification.
Standardized numerical features using scale().

2. Principal Component Analysis (PCA)

Automatic PCA with prcomp():
Computed principal components and variance explained.
Generated scree plot and cumulative variance plots.
Visualized the first two PCs with a scatter plot colored by wine quality.

Manual PCA (Eigen Decomposition):
Computed covariance matrix.
Extracted eigenvalues (variance explained) & eigenvectors (PC loadings).
Verified consistency with automatic PCA results.

3. Linear Discriminant Analysis (LDA)

Multiclass LDA:

Modeled wine quality across all categories.
Extracted discriminant coefficients and functions.
Evaluated classification accuracy with a confusion matrix.

Binary Fisher’s LDA:

Compared two selected quality groups (e.g., quality 5 vs 6).
Projected data onto discriminant axis.
Visualized using scatter plots & histograms.

📈 Results & Visualizations

PCA

Variance Explained:

PC1 explained ~28.7% variance, PC2 ~17.1%, PC3 ~14.3%.
First 5 PCs captured ~80% of total variance.

Key Visualizations:
Scree plots (automatic & manual) showed clear elbow after ~5 PCs.
Scatter plot of PC1 vs PC2 revealed clustering patterns by quality.

LDA

Multiclass LDA:
Extracted 5 discriminant functions.
Accuracy moderately improved over random guessing (~best for mid-quality wines).

Binary Fisher’s LDA (5 vs 6):
Clear separation on discriminant axis.
Accuracy higher than multiclass scenario.

Visualizations:

Projection scatter plots and histograms demonstrated separation across quality groups.

▶️ How to Run the Code
Install R (≥ 4.0.0) from CRAN

Install required libraries:
install.packages(c("ggplot2", "MASS"))
Load dataset: Update the CSV path in the script:

wine <- read.csv("C:/Users/Lenovo/Downloads/WineQT.csv")

Run scripts step by step:

Data preprocessing

PCA (automatic & manual)

LDA (multiclass & binary)

View outputs:

Scree plots, PCA scatter plots, LDA projections
Accuracy metrics and confusion matrix

🌍 Real-life Applications

Wine Industry: Identify quality-driving chemical properties for better production and quality assurance.
Customer Segmentation: Classify consumer preferences based on product attributes.
General Use Cases: PCA/LDA methods are widely applied in finance, healthcare, image recognition, bioinformatics, and NLP for dimensionality reduction and classification.
