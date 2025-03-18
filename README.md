# Body Fat Prediction using various regression techniques

## Project Overview

This project performs an end-to-end machine learning analysis to predict body fat percentage using various regression models. The goal is to compare the performance of multiple regression techniques on a body fat dataset, providing insights into which model performs best for this task. The project uses a dataset with features such as age, weight, height, and various body measurements to predict the target variable, body fat percentage.

### Key Features
- **Models Compared**: Linear Regression, Ridge Regression, Lasso Regression, Decision Tree Regression, Random Forest Regression, Gradient Boosting Regression, LightGBM Regression, and Stochastic Gradient Descent (SGD) Regression.
- **Evaluation Metrics**: Mean Squared Error (MSE), R2 Score, and 5-fold Cross-Validation R2 scores (mean and standard deviation).
- **No Outlier Removal**: The raw dataset is used without filtering outliers to preserve all data points.
- **Visualization**: Bar plots for R2 scores and cross-validation results to facilitate model comparison.

### Tools and Libraries
- **Python**: Core programming language.
- **Jupyter Notebook**: Interactive environment for development and visualization.
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, LightGBM.

## Dataset

The project uses an example dataset from [this GitHub repository](https://raw.githubusercontent.com/selva86/datasets/master/BodyFat.csv), which contains body fat percentage and related measurements. You can replace it with your own dataset by updating the `url` variable in the notebook.

- **Target Variable**: `BodyFat` (body fat percentage).
- **Features**: Age, Weight, Height, Neck, Chest, etc.

## Project Structure

- `body_fat_prediction.ipynb`: Jupyter Notebook containing the complete project workflow.
- `README.md`: This file, providing project documentation.

## Setup Instructions

### Prerequisites
- Python 3.x
- Jupyter Notebook or JupyterLab
- Git (for cloning the repository)

## Installation
1. **Clone the Repository**:
   git clone https://github.com/your-username/body-fat-prediction.git
   cd body-fat-prediction
2. **Install Dependencies**: Install the required Python libraries using pip:
   pip install numpy pandas matplotlib seaborn scikit-learn lightgbm jupyter
3. **Launch Jupyter Notebook**

## Project Workflow
1. **Load and Explore Data**: Load the dataset and display basic information (info, head, missing values).
2. **Preprocessing**: Define features and target; no outlier removal is applied.
3. **Data Splitting and Scaling**: Split into train/test sets and standardize features.
4. **Model Training and Evaluation**: Train eight regression models, compute metrics, and perform cross-validation.
5. **Results Visualization**: Display a table and plots comparing model performance, and identify the best model.

## Sample Output
- **Results Table**: A Pandas DataFrame with MSE, R2 Score, CV R2 Mean, and CV R2 Std for each model.
- **R2 Score Plot**: Bar chart comparing R2 scores across models.
- **Cross-Validation Plot**: Bar chart with error bars showing CV R2 means and standard deviations.
- **Best Model**: Printed statement identifying the top model (e.g., "Best Model: Random Forest Regression with R2 = 0.85").

## Contributing
Feel free to fork this repository, make improvements, and submit pull requests. Suggestions for enhancements include:

- Adding hyperparameter tuning (e.g., GridSearchCV).
- Including additional regression models.
- Implementing feature selection techniques.

## License
- This project is licensed under the MIT License.

## Acknowledgments
- Dataset source: Kaggle.
- Built with inspiration from machine learning best practices and community resources.
