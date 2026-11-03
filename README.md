# SberAvtopodpiska: ML Conversion Prediction Service

Machine Learning service to predict whether a user session results in a target action conversion for the "SberAvtopodpiska" car subscription service.  
Model: Random Forest with balanced class weights.
Key features: UTM parameters, visit activity, device info.

## Project Structure
- README.md                  # This file
- instructions_ru.txt        # Instructions for running the service (in Russian)
- main.py                    # FastAPI web service for real-time predictions
- pipeline.pkl               # Trained ML pipeline
- sber_autopodpiska.ipynb    # Jupyter notebook with EDA, feature engineering, and model training

### Problem Statement
The product team is unable to manually identify users who are likely to submit an inquiry or request a callback.

### Solution
Build a model that, based on visit data, predicts whether a user will perform a target action - 1 or not - 0. The ML pipeline automates the identification of high‑intent users by outputting the probability of conversion and a binary label, enabling the team to prioritize outreach efforts.

- **Task type**: binary classification.
- **Data**: the dataset is provided by the Skillbox platform and is not included in this repository. The data consists of information about user visits and events (hits) that occurred within each session.


### Workflow stages
- Loading and EDA
- Data cleaning
- Target variable creation: a binary target feature was generated for each session based on a list of target events. The conversion rate was ~2.7%, indicating a significant class imbalance.
- Feature engineering
- Modeling: a Random Forest Classifier was chosen as the primary model, as it is robust to outliers, handles encoded categorical features well, and allows for the assessment of feature importance. Hyperparameter tuning was performed using GridSearchCV.
- Model evaluation: ROC-AUC on the test set: 0.697
- The difference between the train and test ROC-AUC scores was ~0.02, suggesting that no overfitting occurred.
- <img width="1240" height="924" alt="roc" src="https://github.com/user-attachments/assets/4a326136-f9a8-4cc0-80e1-b26a94f9823f" />






  



