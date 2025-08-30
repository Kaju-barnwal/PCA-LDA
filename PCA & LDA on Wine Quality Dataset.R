# Load libraries
library(ggplot2)
library(MASS)

# Read CSV 
wine <- read.csv("C:/Users/Lenovo/Downloads/WineQT.csv")

# Check structure
str(wine)
summary(wine)
colnames(wine)
head(wine)

# Standardize features
wine_features <- subset(wine, select = -c(Id, quality))
wine_scaled <- scale(wine_features)

#========================================================
# Principal Component Analysis (PCA) on Wine Dataset
#========================================================

# Fit PCA using standardized features
pca_fit <- prcomp(wine_scaled, center = TRUE, scale. = TRUE)
pca_fit

#--------------------------------------------------------
# Variance explained by each principal component (PC)
#--------------------------------------------------------
pve_pca <- (pca_fit$sdev^2) / sum(pca_fit$sdev^2)   # Proportion of variance explained
pve_pca
cumsum(pve_pca)                                     # Cumulative variance explained

# Scree plot of variance explained
scree_plot_auto <- qplot(1:length(pve_pca), pve_pca, geom="line") + 
  geom_point() +
  xlab("Principal Component") +
  ylab("Proportion of Variance Explained") +
  ggtitle("Scree Plot - PCA (Wine Dataset)")
print(scree_plot_auto)

# Cumulative variance
cum_var_pca <- cumsum(pve_pca)
cum_var_pca

#--------------------------------------------------------
# PCA scatter plot (first two PCs)
#--------------------------------------------------------
pca_scores <- data.frame(pca_fit$x, Quality = as.factor(wine$quality))

ggplot(pca_scores, aes(PC1, PC2, color = Quality)) +
  geom_point(size=2, alpha=0.7) +
  ggtitle("PCA Projection of Wine Data (PC1 vs PC2)") +
  xlab("Principal Component 1") +
  ylab("Principal Component 2")

#========================================================
# Manual PCA Computation (via Eigen Decomposition)
#========================================================

# Covariance matrix of standardized data
cov_mat <- cov(wine_scaled)
cov_mat

# Eigen decomposition of covariance matrix
eigen_decomp <- eigen(cov_mat)
eigen_decomp

# Extract eigenvalues (variances explained) and eigenvectors (PC directions)
eig_values <- eigen_decomp$values
eig_vectors <- eigen_decomp$vectors

# Proportion of variance explained
pve_manual <- eig_values / sum(eig_values)
pve_manual

# Scree plot (manual PCA)
scree_plot_manual <- qplot(1:length(pve_manual), pve_manual, geom="line") + 
  geom_point() + 
  xlab("Principal Component") + 
  ylab("Proportion of Variance Explained") + 
  ggtitle("Scree Plot - PCA (Manual Computation)") + 
  theme_minimal()
print(scree_plot_manual)

# Cumulative variance (manual PCA)
cum_var_manual <- cumsum(pve_manual)
cum_var_manual

#========================================================
# Linear Discriminant Analysis (LDA) on Wine Dataset
#========================================================

# Convert target variable to factor (required for classification)
wine$quality <- as.factor(wine$quality)

# Remove ID column (not useful for modeling)
wine <- subset(wine, select = -Id)
head(wine)

#-------------------------------
# Multiclass LDA (all quality levels)
#-------------------------------

# Fit LDA model
lda_fit <- lda(quality ~ ., data = wine)

# Predict classes and discriminant scores
lda_pred <- predict(lda_fit)

# Discriminant coefficients (linear combinations of features)
print(lda_fit$scaling)

# First few discriminant scores (LD1, LD2, etc.)
head(lda_pred$x)

# Confusion matrix and accuracy
conf_matrix <- table(Predicted = lda_pred$class, Actual = wine$quality)
print(conf_matrix)
cat("Overall Accuracy:", mean(lda_pred$class == wine$quality), "\n")

# Scatter plot of first 2 linear discriminants
lda_data <- data.frame(lda_pred$x, quality = wine$quality)
ggplot(lda_data, aes(LD1, LD2, color = quality)) +
  geom_point(size=2, alpha=0.7) +
  ggtitle("Multiclass LDA Projection of Wine Quality Data")

#========================================================
# Fisher’s LDA (Binary Classification Example: Quality 5 vs 6)
#========================================================

# Select only two classes
wine_binary <- subset(wine, quality %in% c("5", "6"))

# Separate features and labels
X <- as.matrix(subset(wine_binary, select = -quality))
y <- wine_binary$quality

# Split into class-specific groups
X_class5 <- X[y=="5", ]
X_class6 <- X[y=="6", ]

# Compute class means
mu5 <- colMeans(X_class5)
mu6 <- colMeans(X_class6)

# Compute within-class scatter matrix (pooled covariance)
S1 <- cov(X_class5)
S2 <- cov(X_class6)
Sw <- S1 + S2

# Fisher’s discriminant vector
w_vec <- solve(Sw) %*% (mu5 - mu6)

# Project data onto discriminant vector
proj_class5 <- X_class5 %*% w_vec
proj_class6 <- X_class6 %*% w_vec

# Combine projections into a dataframe for plotting
proj_df <- data.frame(
  projection = c(proj_class5, proj_class6),
  quality = factor(c(rep("5", nrow(X_class5)), rep("6", nrow(X_class6))))
)

# Scatter plot of projections (1D LDA)
proj_df$index <- 1:nrow(proj_df)
ggplot(proj_df, aes(x=index, y=projection, color=quality)) +
  geom_point(alpha=0.7) +
  ggtitle("Fisher's LDA Projection: Wine Quality 5 vs 6") +
  ylab("LDA Projection") + xlab("Sample Index")

# Output discriminant vector
w_vec


