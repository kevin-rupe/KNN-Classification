# KNN-Classification
Project completed in pursuit of Master's of Science in Data Analytics.

## PART I: RESEARCH QUESTION


### PROPOSAL OF QUESTION 

How effective can KNN (K-Nearest Neighbors) predict patient readmissions to the hospital given correlated medical factors, so that the medical facility can take appropriate action to reduce readmission rates? 

### DEFINED GOAL

The primary goal of this analysis is to develop a machine learning model using KNN to help the medical facility identify patients who are more at risk for readmission. 


## PART II: METHOD JUSTIFICATION


### EXPLANATION OF CLASSIFICATION METHOD

For this analysis, I chose to use the KNN classification method. KNN stands for K-Nearest Neighbor. It is an algorithm that is used in machine learning to classify observations by looking at how similar (i.e., how near) data points are to similar values. (Srivastava, 2024) 

The dataset is split up into two or more subsets for analysis. For my project, I am only splitting this dataset into two: a training set and a test set. The training set is split by using 70% of the original observations, while the test set uses just 30%. The training set data is then ran through the KNN Classifier which then learns the algorithm for our data. 

Next, the test set data is then used to predict the outcomes using the KNN algorithm that was defined by the training set. After performing this step, we can pull statistical data that measures the accuracy and precision of our model. We can also visualize using a plot function by graphing the ROC (Receiver Operating Characteristic) curve and then identifying the AUC (Area Under the ROC Curve) which shows how well our model performs. 

The expected outcome of this analysis is that we will have all observations classified into two groups: those patients who are predicted to have been readmitted, and those who have not. This analysis will also provide the model’s performance which will give us the certainty at which this model is accurate. A low accuracy will not give us the confidence to recommend any course of action to the medical facility, however, I expect that with the given dataset, we will be able to secure a high model performance for this analysis. 

### SUMMARY OF METHOD ASSUMPTION

One assumption to KNN is that based on distance, a data point is classified as the same as another similar data point. This could be a false assumption as there’s no perfect science to really determining if two things are exactly alike based on how close they are. Also, given various sizes of k, you could easily predict different outcomes for the same data point. However, we can test the model’s performance, so we can have a good level of confidence in the end (Vishalmendekarhere, 2021).

### PACKAGES OF LIBRARIES LIST

![IMG_1616](https://github.com/user-attachments/assets/e4840c78-87ac-48ec-b469-f6cf6485330c)

## PART III: DATA PREPARATION

### DATA PREPROCESSING

One important preprocessing goal is to convert the categorical variables to numeric values using dummy variables. The KNN Classifier method only works with numerical variables, so in order to use categorical variables you have to convert, for example, responses such as ‘Yes’ and ‘No’ to 1 or 0. 

In regression, we need to use k-1 dummies for each variable to counter against multicollinearity, but in machine learning algorithms such as KNN, we need to keep all dummies because we don’t need to assume that there is a linear relationship between our target and feature variables. (Shmueli, 2015) 

### DATA SET VARIABLES 

![IMG_1617](https://github.com/user-attachments/assets/d5e572a2-15e7-41be-a631-d3ebb1718198)


### STEPS FOR ANALYSIS

First, I need to import the Pandas library and load the dataset into Jupyter Notebook (Pandas, 2023).
```python
#import CSV file
import pandas as pd
med_df = pd.read_csv("C:/Users/e0145653/Documents/WGU/D209 - Data Mining/medical_clean.csv")
```

Next, I will view the data to get a sense of what variables we have in our dataset by viewing the dataframe’s first few rows. There appear to be some variables here that I will not need for this analysis; that are useless based on my research question. 
```python
#view dataset
med_df.head()
```
![IMG_1618](https://github.com/user-attachments/assets/9f2184ef-26e9-4100-b59e-1d96425fe0bb)

After evaluating all of the variables, I decide on dropping the variables that I feel are not significant to my research question. 
```python
## drop unused columns
med_df = med_df.drop(['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 'State',
                      'County', 'Zip', 'Lat', 'Lng', 'Population', 'Area', 'TimeZone', 
                      'Job', 'Children', 'Age', 'Income', 'Marital', 'Gender', 'VitD_levels', 
                      'Doc_visits', 'Full_meals_eaten', 'vitD_supp', 'Soft_drink', 
                      'TotalCharge', 'Additional_charges', 'Item1', 'Item2', 'Item3', 
                      'Item4','Item5', 'Item6', 'Item7', 'Item8'], axis=1)
```


Next, I will check for duplicated rows and for missing values in the dataset. I find none, so there is no need to clean the data for these issues. 
```python
#check for duplicated/missing values
print(med_df.duplicated().value_counts())
print("")
print('Variables        Missing Values')
print('---------        --------------')
print(med_df.isna().sum())
```
![IMG_1619](https://github.com/user-attachments/assets/93b7c81d-c919-4142-ae2e-9b82c8e572af)


Next, I need to verify if there are any outliers in the variable Initial_days since it’s a continuous numeric variable. The best way to check for outliers is to use a boxplot which will display outliers on the outside of each of the whiskers. (Waskom, 2012-2022) 

```python
#check for outliers
import seaborn as sns
sns.boxplot(med_df.Initial_days, orient='h').set(title='Initial_days')
```
![IMG_1620](https://github.com/user-attachments/assets/0de00600-df23-48e4-8c83-818b713e4941)


Before performing any machine learning, we need to convert all of our categorical variables to dummy variables. For all of the variables with “Yes” or “No” values, I used one-hot encoding method of replacing “Yes” with 1, and “No” with 0. 

```python
#One-Hot Encoding for (Yes/No) variables
prefix_list1 = ['ReAdmis','HighBlood', 'Stroke', 'Arthritis', 
               'Diabetes', 'Anxiety', 'Asthma',
               'Overweight', 'Hyperlipidemia', 'BackPain',
               'Allergic_rhinitis', 'Reflux_esophagitis']

prefix_dict = {'Yes': 1, 'No': 0}

for col in prefix_list1:
    med_df[col] = med_df[col].replace(prefix_dict)
```

For the other variables with more categorical values, I used Pandas get_dummies function to convert these to 1’s and 0’s. To make these variable names shorter, I renamed each one in the step of get_dummies, and then renamed them one time further to remove white spaces. 
```python
#Get dummies for variables
ia = pd.get_dummies(med_df['Initial_admin'], prefix='IA', prefix_sep='_', drop_first=False)
cr = pd.get_dummies(med_df['Complication_risk'], prefix='CompRisk', prefix_sep='_', drop_first=False)
svc = pd.get_dummies(med_df['Services'], prefix='Svc', prefix_sep='_', drop_first=False)

#concat dataframes
med_df = pd.concat([med_df, ia, cr, svc], axis=1)

#drop former variables
med_df = med_df.drop(['Initial_admin', 'Complication_risk', 'Services'], axis=1)
```
I would like to continue to evaluate the variables to determine which ones are best suited to my target variable. For this step, I am going to use a feature selection method called SelectKBest from scikit-learn which will evaluate all the variables and can provide me with the best variables based on significant p-values (i.e., < 0.05). 

```python
X = med_df.drop(['ReAdmis', 'Initial_days'],1)
y = med_df['ReAdmis']

from sklearn.feature_selection import SelectKBest, f_classif
skbest = SelectKBest(score_func=f_classif, k='all')

X_new = skbest.fit_transform(X,y)

p_values = pd.DataFrame({'Feature': X.columns,
                        'p_value': skbest.pvalues_}).sort_values('p_value')
#p_values[p_values['p_value']<0.5]

features_to_keep = p_values['Feature'][p_values['p_value']<0.5]
print(features_to_keep)
```
![Image 3-8-25 at 11 00 AM](https://github.com/user-attachments/assets/a9580347-e577-4559-8ec8-858365c02ee6)


## PART IV: ANALYSIS


### SPLITTING THE MODEL
```python
#split the data into train/test sets
from sklearn.model_selection import train_test_split

X = med_df.drop(['ReAdmis'], axis=1)
y = med_df['ReAdmis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16, stratify=y)
```

### OUTPUT AND INTERMEDIATE CALCULATIONS

The technique I am using for this project is the machine learning algorithm KNN classification method. This method groups similar data points together based on their k-nearest neighbor. In order to properly use this method, I had to set what ‘k’ would be in the model. I used hyperparameter tuning via the GridSearchCV function in scikit-learn. This function tests which parameters fit best in my model. The parameters that were determined were n_neighbors (i.e., what ‘k’ should be), metric, and weights. I input k-values into a dictionary for GridSearchCV to use starting at 5 through 180 incrementing each step by a factor of 10. This provided the best value for k of 35 (Jordan, 2017). You can also visualize this using Matplotlib.pyplot, which is shown below. 

![IMG_1621](https://github.com/user-attachments/assets/c48dd29f-f67b-41d9-8d98-c0dd2d19e395)

The best method for the weights parameter is uniform, which equally weighs all points in each neighborhood. The metric parameter determines the best method to use for computing distance. There are quite a few options to use, but I chose to just stick with 3 different methods to test: Manhattan distance, Euclidean distance, and Minkowski distance. The best method, chosen by GridSearchCV function, is Manhattan distance. The calculations and formula for this method are shown below. The weights parameter is used in prediction (Scikit-learn, 2007-2024).

![IMG_1622](https://github.com/user-attachments/assets/f628a65f-c21e-46e5-a95b-b8cc472118d6)

> The GridSearchCV accuracy returns a 98.3% score which assures me that these parameters are well fit for our model.

### CODE EXECUTION
```python
#Grid Search to find best parameters
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np

kf = KFold(n_splits=10, shuffle=True, random_state=42)
param_grid = {'n_neighbors': np.arange(5,180,10),
             'metric': ('manhattan', 'euclidean', 'minkowski'),
             'weights': ('uniform', 'distance')}

knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=kf)
knn_cv.fit(X_train, y_train)

print('Best Parameters: {}'.format(knn_cv.best_params_))
print('Best Score: {}'.format(knn_cv.best_score_))
```
> Best Parameters: {'metric': 'manhattan', 'n_neighbors': 35, 'weights': 'uniform'}
>
> Best Score: 0.9825714285714285

```python
## KNN model with best parameters

#knn model
knn = KNeighborsClassifier(metric='manhattan',
                          n_neighbors=35,
                          weights='uniform')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

#confusion_matrix & classification_report
best_matrix = confusion_matrix(y_test, y_pred)
print(best_matrix)
print(classification_report(y_test, y_pred))

# Extract TN, TP, FN and FP from the initial KNN model 
TN = best_matrix[0,0]
TP = best_matrix[1,1]
FN = best_matrix[1,0]
FP = best_matrix[0,1]

# Calculate and print the accuracy
accuracy = (TN + TP) / (TN + FN + FP + TP)
print("The accuracy    of the new model is", np.round((accuracy * 100),2), "%")

# Calculate and print the sensitivity
sensitivity = TP / (TP + FN)
print("The sensitivity of the new model is", np.round((sensitivity * 100),2), "%")

# Calculate and print the specificity
specificity = TN / (TN + FP)
print("The specificity of the new model is", np.round((specificity * 100),2), "%")

y_pred_proba = knn.predict_proba(X_test)[:,1]

#New ROC curve
y_pred_proba = knn.predict_proba(X_test)[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba)

#New Plot
plt.plot([0,1], [0,1], 'k--')
plt.plot(false_positive_rate, true_positive_rate, label='KNN')
plt.xlabel('False + Rate')
plt.ylabel('True + Rate')
plt.title('KNN ROC Curve')
plt.show()

#New AUC
auc = roc_auc_score(y_test, y_pred_proba)
print('AUC score of the new model is', np.round((auc * 100),2), '%')
```
![Image 3-8-25 at 11 09 AM](https://github.com/user-attachments/assets/a0c5b945-a43c-41ad-993e-1852b8ac37d9)


## PART V: DATA SUMMARY AND IMPLICATIONS


### ACCURACY AND AUC

There are many metrics in scikit-learn that we can use to test the model’s performance. The confusion matrix can help us determine the accuracy, sensitivity, specificity, precision, recall, and F1. Scikit-learn’s classification_report uses the confusion matrix to automatically calculate these values. Accuracy is the total true positives and negatives over the total number of samples. This model’s accuracy is 97.6%. Precision is the true positives compared to both true and false positives. This model’s precision rate is 98%. Recall gives us the measurement of true positives divided by the number of true positives and false negatives. This model’s recall rate is also 98%. The F1-score provides the average of precision and recall, which is again 98%. 

The accuracy of the training set compared to the testing set is derived by using the .score( ) from the GridSearchCV function. Both accuracies are around 98% accurate. We can use the AUC (Area under the Curve) metric to determine our the model’s performance. To visualize this better, we can plot the ROC curve. The ROC (Receiver Operator Characteristic) curve plots the true positive rate versus the false positive rate (Bhandari, 2024). We then can find the AUC which tells how well the model accurately makes predictions. The graph below shows a 99.84% AUC which is a nearly perfect model. 

![IMG_1623](https://github.com/user-attachments/assets/76817d73-1739-409e-b450-b9fb61060636)

> AUC score of the new model is 99.84%.

### RESULTS AND IMPLICATIONS

The results of this KNN machine learning algorithm were extremely accurate. With metrics in the high 90 percent ranges, there is strong confidence in its ability to accurate predict outcomes. Specifically in my model, I answered the question of how accurately KNN could predict patient readmissions. Using SelectKBest, we learned that medical conditions of asthma, chronic back pain, arthritis, obesity all played a significant role in patient readmission. Also, services received such as CT Scan, MRI and IV’s played a larger role than patients just having routine blood work. Patients admitted via the Emergency Room, having elective procedures, and being kept for observation also all played a significant part in a patient being readmitted. Given these factors, we can accurately predict whether a patient will be readmitted by 99.84% (AUC). In my initial KNN model, I did not use hyperparameter tuning, and the model also performed well (AUC = 99.5%), but with hyperparameter tuning it did improve some, but only slightly. 

### LIMITATION
  	
KNN is sensitive to imbalanced datasets. In the medical dataset I was using, there was not an equal split of patient readmissions. Only 37% of patients were readmitted. Given this disparity, using KNN is disadvantaged in that the model might assign more patients as not being readmitted, and therefore, could improperly classify the results. 

### COURSE OF ACTION

The recommended course of action for the medical facility would be to use the medical factors used in this model to determine whether a patient will be readmitted or not. The medical staff should be consulted for ways the patients could be treated more effectively for these medical conditions, services they received, and even the means by which the patient was initially admitted to the hospital. Given more precise care for these certain factors, the hospital could certainly lower their patient readmission rate. This would be a positive move not only for the overall well-being of their patients, but also improve the company’s financial position. 	

## PART VI: SUPPORTING DOCUMENTATION

#### SOURCES FOR THIRD-PARTY CODE

Pandas (2023, June 28). Retrieved September 27, 2023, from https://pandas.pydata.org/docs/reference/index.html.

Waskom, M. (2012-2022). Seaborn Statistical Data Visualization. Retrieved September 27, 2023, from https://seaborn.pydata.org/index.html.

Scikit Learn (2007-2024). scikit-learn: Machine Learning in Python. Retrieved March 6, 2024, from https://scikit-learn.org/stable/index.html. 

#### SOURCES 

Srivastava, T. (2024, Jan 04) A Complete Guide to K-Nearest Neighbors (Updated 2024). Retrieved March 5, 2024, from https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/ 
