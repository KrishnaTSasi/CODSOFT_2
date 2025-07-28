# CODSOFT_2
 **Credit Card Fraud Detection Project**
 
This project focuses on detecting fraudulent credit card transactions using an imbalanced dataset. Three machine learning models were trained:

* Logistic Regression
* Decision Tree
* Random Forest
 ✅ Project Procedure Steps

 🧩 1. **Understand the Dataset**

* File: `train.csv`
* Target: `is_fraud` (0 = legit, 1 = fraud)
* Features: mix of numerical, categorical, and geographic data

 🧹 2. **Data Cleaning**

* Drop unnecessary columns:

  * `Unnamed: 0`, `trans_num`, `cc_num`, `first`, `last`, `dob`, `street`, `trans_date_trans_time`
* Drop missing values:

  ```python
  df.dropna(subset=['is_fraud'], inplace=True)
  df.dropna(inplace=True)
  ```

 🔠 3. **Encoding Categorical Features**

* **Label Encoding**:

  * `merchant`, `gender`, `state`, `city`, `job`
* **One-hot Encoding**:

  * `category` (drop first column to avoid dummy trap)

 📏 4. **Feature Scaling**

* Use `StandardScaler` for:

  * `amt`, `lat`, `long`, `merch_lat`, `merch_long`, `city_pop`

 🧪 5. **Train-Test Split**

* Use stratified split to preserve class distribution:

  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
  ```

# 🤖 6. **Model Building (No SMOTE)**

* Use original class imbalance with:

  * `class_weight='balanced'` to compensate for fraud rarity
* Models to build:

  * **Logistic Regression**
  * **Decision Tree**
  * **Random Forest**

 📊 7. **Model Evaluation**

* **Accuracy** – overall correct predictions
* **Precision** – how many predicted frauds are actually fraud
* **Recall** – how many actual frauds are correctly predicted
* **F1-score** – balance between precision and recall
* **Confusion Matrix** – visual representation

🖼️ 8. **Visualize Class Distribution**

```python
sns.countplot(x='is_fraud', data=df)
plt.title("Distribution of Legit vs Fraudulent Transactions")
plt.show()
```

🧪 9. **Testing and Result Export**

* Run the full script:

  ```bash
  python scripts/train_model.py
  ```
* View outputs in `outputs/model_reports/`

** Interpretation**

Each model achieved the **same accuracy** on the test dataset:

 ✅ Model Accuracy

| Model               | Accuracy (%) |
| ------------------- | ------------ |
| Logistic Regression | 99.48        |
| Decision Tree       | 99.48        |
| Random Forest       | 99.48        |

 ⚠️ Interpretation of Results

While these accuracy scores appear impressive, they are **potentially misleading** due to the nature of the dataset:

* The dataset is **highly imbalanced**, with very few fraudulent transactions compared to legitimate ones.
* A naive model that predicts every transaction as "legitimate" would still achieve very high accuracy.

### 📌 Why Accuracy Isn't Enough

In fraud detection, we care more about:

| Metric        | Description                                           |
| ------------- | ----------------------------------------------------- |
| **Precision** | How many of the predicted frauds are actually frauds? |
| **Recall**    | How many actual frauds did the model catch?           |
| **F1-score**  | Harmonic mean of Precision and Recall                 |


 ✅ Conclusion

Even though all models achieved 99.48% accuracy, relying solely on accuracy in a fraud detection scenario is not recommended. Instead, focusing on **recall and F1-score** is essential to ensure that fraudulent transactions are properly identified.



CODSOFT/
├── data/                  # Raw dataset
├── notebooks/             # Jupyter notebook
├── scripts/               # Modular Python scripts
├── outputs/               # Model reports and figures
├── README.md
├── requirements.txt
└── .gitignore

