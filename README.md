# 🩺 Chronic Kidney Disease (CKD) Prediction

## 📌 Overview  
This project aims to predict whether a patient is suffering from **Chronic Kidney Disease (CKD)** using multiple machine learning algorithms.  
We implemented **Decision Tree**, **K-Nearest Neighbors (KNN)**, and **Random Forest** classifiers, evaluated their performance, and compared results with and without feature selection.  
**Random Forest** consistently achieved the best accuracy in both scenarios.

---

## 🎯 Objectives  
- Predict CKD using patient medical records.  
- Compare multiple classification algorithms.  
- Identify the most important features affecting CKD prediction.  

---

## 📂 Dataset  
- **Source**: CKD dataset containing patient health parameters.  
- **Features include**:  
  - Blood Pressure  
  - Blood Urea  
  - Serum Creatinine  
  - Hemoglobin  
  - Age  
  - and more...  
- **Target variable**: CKD or Not CKD  

---

## ⚙️ Project Steps  

### 1️⃣ Data Collection  
- Loaded the CKD dataset containing patient details and lab test results.  

### 2️⃣ Data Preprocessing  
- Handled missing values.  
- Converted categorical data to numerical format.  
- Standardized numeric features.  

### 3️⃣ Feature Selection  
- Applied feature selection to improve accuracy and reduce computation time.  

### 4️⃣ Model Implementation  
- **Decision Tree Classifier**  
- **K-Nearest Neighbors (KNN)**  
- **Random Forest Classifier**  

### 5️⃣ Model Evaluation  
- Metrics used: **Accuracy**, **Precision**, **Recall**, **F1-score**, and **Confusion Matrix**  
- Compared results with and without feature selection.  

### 6️⃣ Results  
- **Random Forest** achieved the highest accuracy in both scenarios.  

---

## 📊 Results Summary  

| Model          | With Feature Selection | Without Feature Selection |
|----------------|-----------------------|---------------------------|
| Decision Tree  | XX%                   | XX%                       |
| KNN            | XX%                   | XX%                       |
| Random Forest  | **Best**              | **Best**                  |

> Replace **XX%** with your actual values.

---

## 📌 Conclusion  
- **Random Forest** is the most effective model for CKD prediction on this dataset.  
- **Feature selection** improved performance and reduced complexity.  

---

## 🛠️ Technologies Used  
- Python 🐍  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  

---

## 📜 License  
This project is licensed under the No License.  

---

