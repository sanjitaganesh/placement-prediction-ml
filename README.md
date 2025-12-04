# Placement Prediction using Machine Learning

This project predicts whether a student will get placed based on academic and activity-related factors using a Logistic Regression model.

---

## Dataset

The dataset used is `Placement_BeginnerTask01.csv` which contains features such as:

- CGPA  
- Placement Training  
- Extracurricular Activities  
- Other student performance indicators  

The target variable is:

- **PlacementStatus** (0 = Not Placed, 1 = Placed)

---

## Workflow

The project follows these steps:

1. **Data Preprocessing**
   - Dropped irrelevant columns
   - Label encoded categorical values
   - Scaled numerical features

2. **Exploratory Data Analysis (EDA)**
   - Visualized placement distribution using bar chart
   - Analyzed CGPA vs Placement relationships using scatter plot

3. **Model Building**
   - Split dataset into training and testing sets
   - Trained a Logistic Regression classifier

4. **Evaluation**
   - Calculated accuracy score
   - Generated confusion matrix
   - Generated classification report

---

## Results

- Achieved **model accuracy greater than 60%**
- Plots saved inside the **`eda_plots/`** folder
- Model performance evaluated using standard metrics

---

## Files in Repository

