# Breast Cancer Diagnosis and Classification Analysis Project

## Project Overview
This project leverages machine learning and deep learning techniques to analyze and classify biomarker data from breast cancer (BC) patients and healthy controls (HC). The code performs data cleaning, feature engineering, model training, evaluation, and visualization, focusing on circular RNAs (e.g., *hsa_circ_0044235* and *hsa_circ_0000250*) and tumor markers (CEA, CA125, CA153) for breast cancer diagnosis. Key functionalities include:
- **Data preprocessing and cleaning**
- **PCA dimensionality reduction and K-means clustering**
- **Training and ensemble of multiple machine learning models** (SVM, Gradient Boosting, Random Forest, Logistic Regression) and a neural network
- **Model performance evaluation** (accuracy, ROC curves, confusion matrices, etc.)
- **Visualization analysis** (clustering plots, feature correlation heatmaps, boxplots, etc.)

## Environment Requirements
To run the code, ensure the following environment is set up:

- **Python Version:** 3.5 or higher
- **Dependencies:**
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - keras
  - tensorflow (as the backend for Keras)

Install the dependencies using the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn keras tensorflow
```

*Note: The code is compatible with Python 3.5 and older versions of Keras and Seaborn.*

## Data Preparation
The project requires the following data files, which should be placed in the project root directory:

- **ROC曲线数据.xlsx (ROC curve data):** Contains biomarker data for healthy controls (HC) and breast cancer patients (BC), including:
  - *hsa_circ_0044235 current/μΑ*
  - *hsa_circ_0000250 current/μΑ*
  - *CEA, CA125, CA153* (optional; missing values will be filled with normal values automatically)

- **BC病理分期.xlsx (BC pathological staging):** Contains pathological staging information for breast cancer patients, including the **病理分期** column (e.g., TNM staging format).

*Data Format:* Excel files with column names matching the feature names in the code.

## Code Structure
The project is organized into the following main modules:

- **Data Loading and Cleaning:** Loads data from Excel files and handles missing or abnormal values.
- **Feature Engineering:** Extracts circular RNA and tumor marker features, followed by standardization.
- **Clustering Analysis:** Applies PCA for dimensionality reduction and K-means for clustering BC and HC samples.
- **Model Training:** Trains SVM, Gradient Boosting, Random Forest, Logistic Regression, and a neural network, then integrates them into a VotingClassifier.
- **Model Evaluation:** Evaluates model performance using cross-validation, ROC curves, confusion matrices, etc.
- **Visualization:** Generates various analytical plots, such as PCA clustering, feature importance, and ROC curves.

## Usage
1. **Prepare Data:** Place `ROC曲线数据.xlsx` and `BC病理分期.xlsx` in the project root directory.
2. **Run the Code:** Execute the following command in the terminal:
   ```bash
   python main.py
   ```
   *(Assuming the code is saved as `main.py`.)*
3. **View Results:** After running, output files will be saved in the project directory, including CSV data files and PNG/SVG image files.

## Results Analysis

Upon completion, the following output files will be generated:

### CSV Files
- `pca_clustering_results.csv`: PCA dimensionality reduction and clustering results.
- `cluster_centers.csv`: K-means clustering centers.
- `roc_curves_data.csv`: Interpolated ROC curve data for each model.
- `roc_curves_raw_data.csv`: Raw ROC curve data.
- `model_performance_summary.csv`: Model performance metrics (accuracy, precision, recall, F1 score, AUC).
- `confusion_matrices.csv`: Confusion matrices for each model.
- `boxplot_data/`: Subdirectory containing boxplot data for each feature (e.g., `boxplot_CEA.csv`).

### Image Files
- `pca_bc_hc_cluster.png`: PCA clustering distribution of BC and HC samples.
- `pca_kmeans_cluster.png`: K-means clustering results.
- `bc_stage_cluster.png`: Clustering analysis of breast cancer patients by TNM staging.
- `feature_correlation.svg`: Feature correlation heatmap.
- `boxplot_*.png`: Boxplots of feature distributions across different breast cancer stages.
- `boxplot_*_comparison.png`: Comparison of tumor marker distributions between BC and HC.
- `boxplot_*_by_stage.png`: Tumor marker distributions across different breast cancer stages.
- `ensemble_confusion_matrix.png`: Confusion matrix for the ensemble model.
- `nn_confusion_matrix.png`: Confusion matrix for the neural network.
- `roc_curves_improved.png`: Improved ROC curves.
- `scatter_*.png`: Feature scatter plots.
- `nn_training_history.png`: Neural network training history (accuracy and loss).
- `model_comparison.png`: Accuracy comparison of different models.
- `feature_importance.png`: Random Forest feature importance.
- `feature_importance_comparison.png`: Feature importance comparison (circular RNA vs. tumor markers).

## Notes
- **Data Completeness:** Ensure that the data files contain the necessary feature columns. If missing, the code will attempt to fill with median values or reference normal values (e.g., CEA filled with 2.5 ng/mL).
- **Sample Size:** A minimum of 10 samples is recommended; too few samples may lead to unreliable model training.
- **Neural Network Training:** If the dataset is insufficient, neural network training may fail, and the code will automatically skip it.
- **Chinese Language Support:** The code is configured for Chinese display on Windows systems (e.g., using SimHei font) to ensure readability of chart titles.

## Contribution and License
- **Contribution:** Contributions are welcome via GitHub issues or pull requests.
- **License:** This project is licensed under the MIT License. See the `LICENSE` file for details.
```

