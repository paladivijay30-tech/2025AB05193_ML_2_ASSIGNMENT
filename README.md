# Heart Disease Prediction System

## Problem Statement

Heart disease remains one of the leading causes of mortality worldwide. Early detection and accurate prediction of heart disease can significantly improve patient outcomes and reduce healthcare costs. This project aims to develop a machine learning-based classification system that predicts the presence of heart disease based on various medical and demographic features.

The goal is to implement and compare multiple classification algorithms to identify the most effective model for heart disease prediction, and deploy an interactive web application for real-time predictions.

## Dataset Description

Dataset Name:Heart Failure Prediction Dataset

Source:Kaggle - Heart Disease Dataset

Dataset Characteristics:
- Total Instances:918
- Number of Features: 11 (12 including target)
- Target Variable: HeartDisease (Binary: 0 = No disease, 1 = Disease present)
- Class Distribution: 
  - No Disease (0): 410 samples (44.7%)
  - Disease (1): 508 samples (55.3%)

Features:
1. Age: Age of the patient (numeric)
2. Sex:Gender of the patient (M/F)
3. ChestPainType:Type of chest pain (TA/ATA/NAP/ASY)
4. RestingBP:Resting blood pressure (mm Hg)
5. Cholesterol: Serum cholesterol (mm/dl)
6. FastingBS:Fasting blood sugar (1 if >120 mg/dl, 0 otherwise)
7. RestingECG:Resting electrocardiogram results (Normal/ST/LVH)
8. MaxHR:Maximum heart rate achieved
9. ExerciseAngina:Exercise-induced angina (Y/N)
10. Oldpeak:ST depression induced by exercise
11. ST_Slope:Slope of peak exercise ST segment (Up/Flat/Down)

## Models Used

### Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8859 | 0.9297 | 0.8716 | 0.9314 | 0.9005 | 0.7694 |
| Decision Tree | 0.7880 | 0.7813 | 0.7890 | 0.8431 | 0.8152 | 0.5691 |
| KNN | 0.8859 | 0.9360 | 0.8857 | 0.9118 | 0.8986 | 0.7686 |
| Naive Bayes | 0.9130 | 0.9451 | 0.9300 | 0.9118 | 0.9208 | 0.8246 |
| Random Forest | 0.8696 | 0.9314 | 0.8750 | 0.8922 | 0.8835 | 0.7356 |
| XGBoost | 0.8587 | 0.9219 | 0.8725 | 0.8725 | 0.8725 | 0.7140 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Demonstrates strong performance with 88.59% accuracy and excellent recall (93.14%), making it highly effective at identifying positive heart disease cases. The model achieves a good balance between precision and recall, with an ROC-AUC of 0.9297 indicating strong discriminative ability. Well-suited for this linearly separable dataset. |
| Decision Tree | Shows the weakest performance among all models with 78.80% accuracy and lowest ROC-AUC (0.7813). The model likely suffers from overfitting on the training data and poor generalization. The MCC score of 0.5691 suggests moderate correlation with actual outcomes. May require pruning or ensemble methods for improvement. |
| KNN | Achieves strong performance (88.59% accuracy) with the second-highest ROC-AUC (0.9360), indicating excellent ability to distinguish between classes. Balanced precision (88.57%) and recall (91.18%) make it reliable for both positive and negative predictions. Performance may be sensitive to the choice of k and distance metric. |
| Naive Bayes | Best performing modelwith highest accuracy (91.30%), ROC-AUC (0.9451), and MCC (0.8246). Excellent precision (93.00%) minimizes false positives, while maintaining good recall (91.18%). The probabilistic approach works exceptionally well for this dataset despite the independence assumption. Recommended for deployment. |
| Random Forest | Shows solid ensemble performance with 86.96% accuracy and strong ROC-AUC (0.9314). Provides good balance across all metrics with an F1 score of 0.8835. Benefits from combining multiple decision trees to reduce overfitting. Feature importance analysis reveals MaxHR, ST_Slope, and Oldpeak as top predictors. |
| XGBoost | Achieves balanced performance (85.87% accuracy) with equal precision and recall (87.25%), resulting in identical F1 score. ROC-AUC of 0.9219 indicates good ranking capability. While competitive, it doesn't outperform simpler models like Naive Bayes, suggesting the dataset may not benefit significantly from gradient boosting complexity. |

## Key Findings

1. Best Model:Naive Bayes achieves the highest overall performance with 91.30% accuracy and 0.9451 ROC-AUC
2. Model Complexity vs Performance:Simpler models (Naive Bayes, Logistic Regression) outperform complex ensemble methods, suggesting the dataset has clear decision boundaries
3. Consistent Top Performers:*Naive Bayes, KNN, and Logistic Regression all achieve >88% accuracy with strong ROC-AUC scores above 0.92
4. Class Balance:All models show good recall (>87%), effectively identifying heart disease cases
5. Feature Engineering Impact:Proper scaling and encoding of categorical variables contributed significantly to model performance

## Project Structure

```
heart-disease-prediction/
│
├── heart_app.py                    # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── Logistic_Regression.pkl         # Trained models
├── Decision_Tree.pkl
├── KNN.pkl
├── Naive_Bayes.pkl
├── Random_Forest.pkl
├── XGBoost.pkl
├── scaler.pkl
├── feature_names.pkl
│
├── test.ipynb                      # Model training notebook
├── heart.csv                       # Dataset
└── model_metrics.json              # Stored metrics
```

## Installation & Usage

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/paladivijay30-tech/2025AB05193_ML_2_ASSIGNMENT.git
cd 2025AB05193_ML_2_ASSIGNMENT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run heart_app.py
```

4. Open your browser and navigate to `http://localhost:8501`

### Using the Application

1. Select Model:Choose from 6 classification models in the sidebar
2. Upload Data:Upload your test dataset (CSV format) with the same features
3. View Results:See predictions, metrics, confusion matrix, and classification report
4. Download:Export predictions with probabilities as CSV

## Deployment

The application is deployed on Streamlit Community Cloudand accessible at:

🔗 **Live App:** [https://2025ab05193ml2assignment-hwgevdyxem692xmu6fcgkr.streamlit.app/]

## Technologies Used


- Scikit-learn: Model implementation and evaluation
- XGBoost:Gradient boosting classifier
- Streamlit:Interactive web application
- Plotly: Interactive visualizations
- Pandas & NumPy: Data manipulation
- Matplotlib: Styling and gradients

## Model Training Details

- Train-Test Split:80-20 split with stratification
- Feature Scaling:StandardScaler for numerical features (fit only on training data)
- Encoding:One-hot encoding for categorical variables
- Cross-Validation: 5-fold CV for model validation
- Random State: 42 (for reproducibility)
- Data Leakage Prevention: Scaling applied after train-test split

## Evaluation Metrics

All models were evaluated using:
- Accuracy:Overall correctness of predictions
- ROC-AUC:Area under the ROC curve
- Precision:Positive predictive value
- Recall:Sensitivity/True positive rate
- F1 Score:Harmonic mean of precision and recall
- MCC:Matthews Correlation Coefficient (accounts for class imbalance)

## Future Enhancements

- [ ] Implement hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- [ ] Add SHAP values for model interpretability
- [ ] Include feature importance visualization
- [ ] Support for real-time single patient prediction
- [ ] Model performance monitoring dashboard
- [ ] API endpoint for integration with other systems

## Author

PALADI S G VENKATA VIJAY
M.Tech (AIML) - BITS Pilani WILP  
Machine Learning - Assignment 2  
Roll Number: 2025AB05193

## Acknowledgments

- Dataset: Kaggle Heart Disease Dataset
- BITS Pilani WILP - Machine Learning Course
- Streamlit Community Cloud for free hosting

## License

This project is created for academic purposes as part of the Machine Learning course assignment.
Note:This application is for educational purposes only and should not be used as a substitute for professional medical diagnosis.
