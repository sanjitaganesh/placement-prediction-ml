# Placement Prediction using Machine Learning

This project predicts whether a student will get placed based on academic performance and extracurricular involvement using a Logistic Regression model.

---

## Dataset

The dataset used is `Placement_BeginnerTask01.csv` and includes features such as:

- CGPA  
- Placement Training  
- Extracurricular Activities  
- Internships and project-related indicators  

The target variable is:

- **PlacementStatus** (0 = Not Placed, 1 = Placed)

---

## Workflow

1. **Data Preprocessing**
   - Removed irrelevant columns such as Student ID
   - Encoded categorical features into numerical format
   - Applied feature scaling to normalize input data

2. **Exploratory Data Analysis (EDA)**
   - Visualized the distribution of placement outcomes
   - Analyzed the relationship between CGPA and placement status

3. **Model Building**
   - Split the dataset into training and testing sets (80:20)
   - Trained a Logistic Regression classifier for binary classification

4. **Evaluation**
   - Evaluated model performance using accuracy score
   - Generated confusion matrix and classification report

---

## Results and Observations

- The model achieved an accuracy of approximately **XX%** on the test dataset  
- Students with higher CGPA show a higher likelihood of being placed  
- EDA plots are saved in the **`eda_plots/`** directory for reference

---

## Files in Repository

- Python script / notebook for model training and evaluation  
- `Placement_BeginnerTask01.csv` dataset  
- `eda_plots/` folder containing visualization outputs
