# Customer Segmentation using K-Means Clustering 📊🤖

This project performs **customer segmentation** using **K-Means clustering** on a marketing dataset. It includes data cleaning, feature engineering, visualization, clustering, and dimensionality reduction using **t-SNE** for intuitive visualization.

---

## 📂 Dataset

* File name: `new.csv`
* The dataset contains customer demographics, purchase behavior, and response data.

> ⚠️ Ensure `new.csv` is present in the project root directory before running the code.

---

## 🛠️ Technologies & Libraries Used

* **Python 3.x**
* **NumPy** – Numerical computations
* **Pandas** – Data manipulation
* **Matplotlib & Seaborn** – Data visualization
* **Scikit-learn** – Machine learning algorithms

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## 🔍 Workflow Overview

### 1️⃣ Data Loading & Inspection

* Load CSV data using Pandas
* Inspect dataset shape
* Check for null values
* View summary statistics

---

### 2️⃣ Data Cleaning

* Identify null values column-wise
* Drop rows containing null values
* Remove unnecessary columns:

  * `Z_CostContact`
  * `Z_Revenue`
  * `Dt_Customer`

---

### 3️⃣ Feature Engineering

* Split `Dt_Customer` into:

  * `day`
  * `month`
  * `year`
* Encode categorical features using **LabelEncoder**

---

### 4️⃣ Exploratory Data Analysis (EDA)

* Count plots for categorical features
* Response-based distribution visualization
* Correlation heatmap (threshold > 0.8)

---

### 5️⃣ Feature Scaling

> K-Means is sensitive to scale. Numerical features are standardized using **StandardScaler**.

---

### 6️⃣ Optimal Cluster Selection (Elbow Method)

* Run K-Means for cluster range **1–20**
* Plot inertia vs number of clusters
* Elbow point observed at:

```text
n_clusters = 5
```

---

### 7️⃣ K-Means Clustering

* Apply K-Means with:

  * `n_clusters = 5`
  * `init = k-means++`
  * `max_iter = 500`

* Generate customer segments

---

### 8️⃣ Dimensionality Reduction using t-SNE

Since the data is high-dimensional:

* Apply **t-SNE** to reduce dimensions to 2
* Visualize clusters in 2D space

---

## 📈 Visualizations

* Categorical feature distributions
* Response-based comparisons
* Correlation heatmap
* Elbow curve
* t-SNE scatter plot with cluster coloring

---

## 📌 Final Output

* Customers segmented into **5 distinct clusters**
* Clear visual separation using t-SNE
* Useful for:

  * Targeted marketing
  * Personalized offers
  * Customer behavior analysis

---

## ▶️ How to Run

```bash
python main.py
```

(or run the notebook cell-by-cell if using Jupyter Notebook)

---

## 📁 Project Structure

```text
├── new.csv
├── main.py
├── README.md
```

---

## ⭐ Key Takeaways

* K-Means is effective for customer segmentation
* Proper preprocessing significantly improves results
* t-SNE helps visualize high-dimensional clusters

---

## 📜 License

This project is for **educational purposes**.

---

Happy Learning & Clustering! 🚀
