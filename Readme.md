# Jordanian Dialect Classification Using Machine Learning and NLP


This repository contains an end-to-end Machine Learning and Natural Language Processing (NLP) pipeline designed to classify four distinct Jordanian sub-dialects: **Ammani, Falahi, Karaki, and Bedouin**. 

This project was developed for the **Pattern Recognition** course at the **University of Jordan**, under the supervision of **Dr. Tamama Al-Sarhan**.

##  Team Members
* **Bayan Marashde** (0233730)
* **Tamerhena Abu-Ghuzleh** (0235703)
* **Malak Anwar** (0234919)
* **Aisha Ramzi** (0235761)

---

##  Project Objectives
The Arabic language has a high degree of morphological variation. Identifying fine-grained dialects within the same country is a complex pattern recognition problem due to massive linguistic overlap. This project aims to:
1. Clean and preprocess informal Arabic dialectal text.
2. Extract deep contextual features using large language models (**MARBERT**).
3. Train, optimize, and evaluate multiple ML classifiers.
4. Provide an interactive live-testing tool for real-time dialect prediction.

---

##  Methodology

### 1. Data Preprocessing
The dataset underwent rigorous cleaning to remove noise from informal text:
* Removal of URLs, English characters, numbers, and emojis.
* Normalization of Arabic letters (e.g., standardizing 'Alef').
* Stripping of diacritics (Harakat) and punctuation.

### 2. Feature Extraction 
Traditional TF-IDF struggles with short, overlapping sentences. We utilized **MARBERT** (a bidirectional Transformer optimized for Arabic dialects) to extract dense 768-dimensional contextual embeddings from the `[CLS]` token of each sentence.

### 3. Model Training & Data Splitting
To ensure robust evaluation and prevent data leakage, we used a highly strict **60-20-20 Stratified Split**:
* **60% Training:** Used to fit the models.
* **20% Validation:** Used for hyperparameter tuning.
* **20% Unseen Testing:** Strictly isolated for final accuracy reporting.

The models trained include:
* Support Vector Machine (**SVM**)
* K-Nearest Neighbors (**KNN**)
* Random Forest (**RF**)
* Multi-Layer Perceptron (**MLP**)

### 4. Optimization & Ensemble Learning
* **Hyperparameter Tuning:** Applied `GridSearchCV` on the SVM to find the optimal regularization parameter (`C`) and kernel type.
* **Voting Classifier:** Combined the predictions of the Tuned SVM, MLP, and Random Forest using a 'soft' voting mechanism to maximize accuracy and minimize individual model bias.

---

##  Results & Visualizations
The optimized Voting Classifier and Tuned SVM demonstrated robust performance, successfully passing the strong baseline accuracy of **77%** on the strictly isolated test set. 

**Visual Diagnostics Included in the Notebook:**
* **PCA 2D Projections:** Visualizing the high-dimensional decision boundaries drawn by the models.
* **Confusion Matrices:** Highlighting misclassifications between geographically adjacent dialects.
* **Error Analysis Summary:** Exploring the impact of linguistic overlap and ultra-short text sequences on misclassifications.

---

## How to Run the Code

The project is built entirely in **Google Colab** for ease of use and GPU acceleration.

1. Clone this repository or download the `.ipynb` notebook file.
2. Open the notebook in [Google Colab](https://colab.research.google.com/).
3. **Enable GPU:** Go to `Runtime` > `Change runtime type` > Select `T4 GPU` (Crucial for MARBERT embeddings).
4. Upload the dataset (`Dialects_Dataset_Final.csv`) and the diagnostic images (`dd1.png`, `dd2.png`, `dd3.png`) to the Colab file explorer on the left panel.
5. Run the cells sequentially. 
6. Scroll down to the **Interactive Live Testing** section to type your own Arabic sentences and see the model's predictions in real-time!

