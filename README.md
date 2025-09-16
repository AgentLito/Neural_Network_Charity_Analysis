🧠 Neural Network Charity Analysis

Deep Learning & Machine Learning Project

📌 Project Overview

This project applies deep learning neural networks to predict whether funding applicants will be successful for the nonprofit Alphabet Soup. Using Python and TensorFlow, we built and optimized a binary classifier to ensure the foundation’s funding is allocated effectively.

🎯 Purpose

Analyze a dataset of 34,000+ organizations that previously received funding.

Preprocess, clean, and encode categorical data for modeling.

Build, train, and optimize a neural network to classify applicants as “successful” or “not successful.”

🛠️ Tech Stack

Language: Python 3.7

Environment: Jupyter Notebook

Libraries: Pandas, Scikit-learn, TensorFlow/Keras

Dataset: charity_data.csv

⚙️ Approach
🔹 Data Preprocessing

Target: IS_SUCCESSFUL (binary outcome).

Features: All other columns (after dropping EIN and NAME).

Cleaning: Encoded categorical variables, removed irrelevant features, binned noisy values, and scaled data using StandardScaler().

🔹 Model Architecture

Layers: 2 hidden layers (200 → 90 neurons).

Activation: ReLU for hidden layers, Sigmoid for output.

Optimizer: Adam.

Loss Function: Binary Crossentropy.

Epochs: 500 (tuned to balance performance and avoid overfitting).

📊 Results

Baseline Accuracy: ~72.4%

Optimized Accuracy: ~76.3%

Loss Score: ~0.61

Optimization Steps

Reintroduced NAME column (bucketed rare categories).

Binned ASK_AMT and categorical values with low frequency.

Increased neurons (200, 90).

Tuned epochs (200 → 500).

Tested adding/removing a 3rd hidden layer (minimal improvement).

✅ Key Takeaways

Achieved target model performance with ~76% accuracy.

Demonstrated end-to-end deep learning pipeline: data preprocessing, model design, training, optimization, and evaluation.

Showed importance of feature engineering and hyperparameter tuning for model improvement.

📂 About

This project demonstrates deep learning model development for real-world decision-making in nonprofit funding. It highlights skills in TensorFlow, preprocessing, and model optimization.


Figure 1: Defining a Model.
![DefiningAModel](https://user-images.githubusercontent.com/91812090/162635004-f51fad7f-a36d-4662-ac7c-d8357f357885.png)
Figure 2: Accuracy After Optimization.
![AccuracyAfter](https://user-images.githubusercontent.com/91812090/162635220-c39ec6b6-be78-4d6f-8526-855a9c281350.png)
Figure 3: Accuracy Before Optimization.
![AccuracyBefore](https://user-images.githubusercontent.com/91812090/162635308-13b9eb5e-c0a2-43aa-9d06-00e7f85bd786.png)
Dataset that we were working on was good fit because of the length of the dataset and its complexity, even though the results weren't the best. 

