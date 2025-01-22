# Credit Card Approval Prediction Model

This project aims to build a machine learning model to predict credit card approval based on a dataset of credit card applications. The project demonstrates data preprocessing, model training using logistic regression, and optimizing the model with hyperparameter tuning using GridSearchCV.

## Steps Involved

### 1. **Data Preprocessing**

Data preprocessing is a critical step before applying machine learning algorithms. In this project, the following steps were implemented:

#### a. **Handling Missing Values:**
   - The dataset contains missing values represented by `?`. These are replaced with `NaN` to identify them easily.
   - **Imputation Strategy:**
     - For **categorical columns** (data type: object), missing values are filled with the most frequent value (mode) in that column.
     - For **numerical columns**, missing values are filled with the mean of the respective column. This ensures that the dataset is complete and ready for modeling.

#### b. **Encoding Categorical Variables:**
   - Categorical variables are transformed into numerical values using **one-hot encoding**. This approach creates a binary column for each category, representing whether the sample belongs to that category.
   - **Drop First**: To avoid multicollinearity (where two features are highly correlated), the first category of each categorical feature is dropped. This reduces the feature space and avoids redundancy.

#### c. **Feature Scaling:**
   - Features are scaled using **StandardScaler**. Standardization ensures that the features have a mean of 0 and a standard deviation of 1. This is important because many machine learning algorithms, including logistic regression, perform better when the features are scaled appropriately.

### 2. **Model Training and Logistic Regression**
   - **Logistic Regression** is used as the classifier for this binary classification task (credit card approval: Yes/No).
   - The model is trained using the preprocessed and scaled training dataset (`X_train` and `y_train`).

### 3. **Hyperparameter Tuning with GridSearchCV**
Hyperparameter tuning is performed using **GridSearchCV** to find the best combination of hyperparameters for the logistic regression model. This process involves the following steps:

#### a. **Hyperparameters Tuned:**
   - **`tol` (Tolerance for stopping criteria)**: This hyperparameter controls the precision of the solution. The algorithm stops iterating when the change between successive iterations is less than this threshold.
   - **`max_iter` (Maximum iterations)**: The maximum number of iterations the solver should run. If the solver doesn’t converge within the specified iterations, it stops.
   
#### b. **Grid Search Process:**
   - A grid of possible values for `tol` and `max_iter` is defined:
     - **`tol`**: [0.01, 0.001, 0.0001]
     - **`max_iter`**: [100, 150, 200]
   - **GridSearchCV** evaluates all possible combinations of these hyperparameters using cross-validation (5-fold). This helps to find the optimal set of hyperparameters that improves the model’s performance.
   - The best combination of parameters is selected based on the model’s cross-validation score.

### 4. **Model Evaluation**
After tuning the model, its performance is evaluated using the following techniques:

#### a. **Confusion Matrix:**
   - The **confusion matrix** helps assess the model's performance by showing the true positives, false positives, true negatives, and false negatives.
   - This matrix gives a better understanding of how well the model classifies the credit card applications, highlighting the types of errors made by the model.

#### b. **Accuracy Score:**
   - The model’s performance is further evaluated using the **accuracy score**, which represents the percentage of correct predictions made by the model on the test dataset. It is calculated as the ratio of correctly predicted instances to the total number of instances.

#### c. **Cross-validation:**
   - The model is cross-validated using **5-fold cross-validation** to ensure that it generalizes well to unseen data. This technique splits the data into 5 parts, uses 4 for training and 1 for testing, and repeats the process 5 times to get an average performance.

### 5. **Conclusion**
   - The final model, optimized using GridSearchCV, predicts whether a credit card application will be approved or not based on several features like loan balances, income levels, and credit history.
   - The project demonstrates the importance of data preprocessing, hyperparameter tuning, and evaluating the model to achieve accurate and reliable predictions.
   - The logistic regression model, after tuning, shows the best performance on the test data, providing insights into credit card approval decisions.

