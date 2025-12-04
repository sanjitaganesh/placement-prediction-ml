# importing the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

# Reading the dataset
dataset = pd.read_csv('Placement_BeginnerTask01.csv')

print("Dataset Loaded Successfully")
print(dataset.head())

# preprocessing
dataset.drop("StudentID", axis=1, inplace=True)  # axis=0 is row and axis=1 is column

from sklearn.preprocessing import LabelEncoder  # labelencoder text with number replacing
le = LabelEncoder()

dataset["PlacementTraining"] = le.fit_transform(dataset["PlacementTraining"])
dataset["PlacementStatus"] = le.fit_transform(dataset["PlacementStatus"])
dataset["ExtracurricularActivities"] = le.fit_transform(dataset["ExtracurricularActivities"])

# Verify datatypes
print("\nData Types after encoding:")
print(dataset.dtypes)

# Splitting the dataset into training set and testing set
X=dataset.drop("PlacementStatus", axis=1)
y=dataset["PlacementStatus"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

#Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create an object of the algorithm / model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)

#Train the model
model.fit(X_train,y_train)
print("\nModel training completed")

# predict the model on testing dataset
y_pred=model.predict(X_test)

#Evaluate accuracy,confusion matrix,classification report
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:\n", cm)

from sklearn.metrics import classification_report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

#Exploratory Data Analysis(EDA)
#Visualising the training set data

#create directory for plots
import os
os.makedirs("eda_plots",exist_ok=True)

#Placement status distribution
dataset["PlacementStatus"].value_counts().plot(kind="bar")
plot.title("Placement Status Distribution")
plot.xlabel("Placemnt Status (0=No, 1=Yes)")
plot.ylabel("Count")
plot.savefig("eda_plots/placement_distribution.png")
plot.show()

#CGPA vs Placement
plot.scatter(dataset["CGPA"],dataset["PlacementStatus"])
plot.title("CGPA vs Placement")
plot.xlabel("CGPA")
plot.ylabel("Placement Status")
plot.savefig("eda_plots/cgpa_vs_placement.png")
plot.show()

# ------------------------------------------
print("\nEDA plots saved to folder: eda_plots/")
print("Notebook execution completed")
# ------------------------------------------


