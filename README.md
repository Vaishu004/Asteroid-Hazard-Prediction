# 🌠 Asteroid Hazard Prediction using Machine Learning

This project uses a Random Forest Classifier to predict whether an asteroid is potentially hazardous, based on NASA's dataset of Near-Earth Objects (NEOs).

Dataset
- `neo_v2.csv`: Contains details like estimated diameter, relative velocity, and miss distance of asteroids.
- Target column: `hazardous` (1 for hazardous, 0 for non-hazardous)

Model Workflow

1. **Data Preprocessing**
   - Selected important numerical features:
     - `est_diameter_min`, `est_diameter_max`, `relative_velocity`, `miss_distance`, `absolute_magnitude`
   - Scaled features using `StandardScaler`

2. **Model**
   - Trained a `RandomForestClassifier` with balanced class weights
   - Predicted hazard probability on test data

3. **Evaluation**
   - Printed:
     - Accuracy Score
     - Confusion Matrix
     - Classification Report

Technologies Used
- Python, Pandas, Scikit-learn


