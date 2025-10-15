# OIBSIP_dataanalytics_04

# 🍷 Task 4: Wine Quality Prediction (Multiclass Classification)

## **Internship**
**Oasis Infobyte** - Data Science Internship (Virtual)

## **Project Goal**
The primary objective of this project was to build and compare multiple machine learning classification models to accurately predict the **quality of wine** based on its physico-chemical properties. This was treated as a **multiclass classification problem** to distinguish between different quality ratings.

## **Dataset**
The project utilized the `WineQT.csv` dataset, which contains 11 input features corresponding to chemical analysis (e.g., fixed acidity, alcohol, volatile acidity) and one target variable (`quality`).

## **Key Steps & Methodology**

1.  **Exploratory Data Analysis (EDA):**
    * Analyzed the distribution of the target variable (`quality` ratings).
    * Visualized feature distributions (e.g., `alcohol`, `pH`).
    * Generated a **Correlation Heatmap** to understand relationships between chemical attributes.
2.  **Data Preprocessing:**
    * Created a binned target variable (`quality_label`) for easier classification.
    * Performed **Stratified Train/Test Split** to ensure balanced representation of all quality classes in the training and testing sets.
3.  **Model Building & Comparison:**
    * Defined and trained three robust classification models:
        * **Random Forest Classifier**
        * **Support Vector Classifier (SVC)**
        * **SGD Classifier**
    * Used Scikit-learn **Pipelines** with **Standard Scaling** for the SVC and SGD models.
4.  **Model Evaluation:**
    * Evaluated performance using **Test Accuracy** and **5-fold Cross-Validation** for robustness.
    * Generated **Classification Reports** and **Confusion Matrices** for detailed analysis of precision, recall, and F1-score for each quality class.
5.  **Feature Importance:**
    * Used the **Random Forest** model to determine and visualize the top chemical features contributing to the quality prediction (e.g., `alcohol`, `volatile acidity`).

## **Tools and Libraries**
* **Python**
* **Pandas & NumPy:** Data manipulation.
* **Matplotlib & Seaborn:** Visualization.
* **Scikit-learn (sklearn):** `train_test_split`, `StandardScaler`, `RandomForestClassifier`, `SVC`, `SGDClassifier`, `cross_val_score`, `classification_report`, `confusion_matrix`.
* **Joblib:** For saving the trained models to disk.

## **Actionable Conclusions**
* **Key Predictors:** **Alcohol content** and **volatile acidity** were identified as the most influential factors in determining wine quality.
* **Model Choice:** [State which model performed best based on your results, e.g., The Random Forest Classifier achieved the highest accuracy and balanced performance across quality classes.]

## **File Structure**
* `wine_quality_prediction.py` - Primary Python script containing the ML pipeline, EDA, and evaluations.
* `WineQT.csv` - The original dataset file.
* `rf_wine_model.joblib` - Saved Random Forest model (output).
* `sgd_wine_pipeline.joblib` - Saved SGD Classifier pipeline (output).
* `svc_wine_pipeline.joblib` - Saved SVC pipeline (output).
* `README.md` - This documentation file.
