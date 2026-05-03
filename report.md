## Healthcare Fraud Detection Using Machine Learning

Healthcare fraud poses a signifigant risk, with estimated financial losses of as much as 15%[<sup>1</sup>](https://google.com). With Canadian healthcare cost projected to be almost 400 billion dollars annually[<sup>2</sup>](https://www.cihi.ca/en/national-health-expenditure-trends/nhex-trends-reports/nhex-trends-2025-snapshot) this could mean as much as a 60 billion dollar loss in 2025 alone.

In attempts to better classify healthcare fraud I have trained a model using machine learning to detect fraudulent cases. We will be using a data set obtained from Kaggle called *Healthcare Fraud Detection Dataset*[<sup>3</sup>](https://www.kaggle.com/datasets/nudratabbas/healthcare-fraud-detection-dataset).

We will be using Panda's to convert our dataset into a frame, and use scikit-learn to train a model. Then we will try some additional tuning to improve model evaluations. Last we will analyze the data and see if we can learn anything about creating better healthcare fraud detection in the future. 

### Set Up

First let's import our dependancies and load the data into a frame using Pandas.


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./healthcare_fraud_detection.csv")

print(df.info())
print(df.describe())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 20 columns):
     #   Column                                 Non-Null Count  Dtype  
    ---  ------                                 --------------  -----  
     0   Provider_ID                            10000 non-null  object 
     1   Claim_ID                               10000 non-null  object 
     2   Patient_Age                            10000 non-null  int64  
     3   Patient_Gender                         10000 non-null  object 
     4   Diagnosis_Code                         10000 non-null  object 
     5   Procedure_Code                         10000 non-null  int64  
     6   Claim_Amount                           10000 non-null  float64
     7   Approved_Amount                        10000 non-null  float64
     8   Insurance_Type                         9650 non-null   object 
     9   Claim_Submission_Date                  10000 non-null  object 
     10  Days_Between_Service_and_Claim         10000 non-null  int64  
     11  Number_of_Claims_Per_Provider_Monthly  10000 non-null  int64  
     12  Provider_Specialty                     9650 non-null   object 
     13  Patient_State                          10000 non-null  object 
     14  Claim_Status                           10000 non-null  object 
     15  Is_Fraud                               10000 non-null  int64  
     16  Length_of_Stay                         10000 non-null  int64  
     17  Visit_Type                             10000 non-null  object 
     18  Chronic_Condition_Flag                 10000 non-null  int64  
     19  Prior_Visits_12m                       9650 non-null   float64
    dtypes: float64(3), int64(7), object(10)
    memory usage: 1.5+ MB
    None
            Patient_Age  Procedure_Code  Claim_Amount  Approved_Amount  \
    count  10000.000000     10000.00000  10000.000000     10000.000000   
    mean      49.755000     86905.21170    572.804406       475.514157   
    std       17.910144     14965.32496    406.202437       323.257165   
    min        1.000000     36415.00000     60.210000        50.350000   
    25%       37.750000     80053.00000    305.205000       257.200000   
    50%       50.000000     93000.00000    461.225000       388.370000   
    75%       62.000000     99213.00000    711.365000       598.347500   
    max       95.000000     99214.00000   6590.700000      4270.890000   
    
           Days_Between_Service_and_Claim  Number_of_Claims_Per_Provider_Monthly  \
    count                    10000.000000                           10000.000000   
    mean                        14.413800                              68.628000   
    std                          8.489875                              14.905872   
    min                          0.000000                              42.000000   
    25%                          7.000000                              60.000000   
    50%                         14.000000                              66.000000   
    75%                         22.000000                              72.000000   
    max                         29.000000                             144.000000   
    
               Is_Fraud  Length_of_Stay  Chronic_Condition_Flag  Prior_Visits_12m  
    count  10000.000000     10000.00000            10000.000000       9650.000000  
    mean       0.082900         2.19930                0.292000          3.026425  
    std        0.275745         1.71046                0.454705          1.722789  
    min        0.000000         0.00000                0.000000          0.000000  
    25%        0.000000         1.00000                0.000000          2.000000  
    50%        0.000000         2.00000                0.000000          3.000000  
    75%        0.000000         3.00000                1.000000          4.000000  
    max        1.000000         5.00000                1.000000         12.000000  
    

### Features and Targets 
Now that we have our data, let's create our features, X, and our target, y. Provider_ID and Claim_ID are not relevant and could skew results, so they will be dropped from our features. We will also drop Is_Fraud as that will be used as our target. 


```python
X = df.drop(columns=["Is_Fraud", "Provider_ID", "Claim_ID"])
y = df["Is_Fraud"]
```

### Cleaning
We can deal with missing data using using fillna and encode the text using get_dummies.


```python
X["Insurance_Type"] = X["Insurance_Type"].fillna("Unknown")
X["Provider_Specialty"] = X["Provider_Specialty"].fillna("Unknown")
X["Prior_Visits_12m"] = X["Prior_Visits_12m"].fillna(0)

X = pd.get_dummies(X, drop_first=True)
```

### Training
Now that our data is ready we can start training the model.


```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;RandomForestClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier(random_state=42)</pre></div> </div></div></div></div>



## Evaluation
Let's take a look at our results.


```python
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
```

    Accuracy: 0.963
                  precision    recall  f1-score   support
    
               0       0.96      1.00      0.98      1847
               1       0.98      0.53      0.69       153
    
        accuracy                           0.96      2000
       macro avg       0.97      0.76      0.83      2000
    weighted avg       0.96      0.96      0.96      2000
    
    

#### Tuning
Results are not too great, recall is at 53%. Let's see if we can improve those results. Let's try making the model more sensitive and flag anything above a 40% confidence rating.


```python
probs = model.predict_proba(X_test)[:, 1]
preds = (probs > 0.4).astype(int)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
```

    Accuracy: 0.9705
                  precision    recall  f1-score   support
    
               0       0.97      1.00      0.98      1847
               1       0.93      0.67      0.78       153
    
        accuracy                           0.97      2000
       macro avg       0.95      0.83      0.88      2000
    weighted avg       0.97      0.97      0.97      2000
    
    

Recall jumped to 67%. It looks like there is potential for finer tuning here. Let's try running a quick loop to find the optimal classification threshold.


```python
from sklearn.metrics import f1_score

best_t = 0
best_score = 0

for t in [i/100 for i in range(10, 90)]:
    preds = (probs > t).astype(int)
    score = f1_score(y_test, preds)

    if score > best_score:
        best_score = score
        best_t = t

print("Best threshold:", best_t)
```

    Best threshold: 0.29
    

### Reevaluation
Let's try evaluating the model again using our best threshold.


```python
probs = model.predict_proba(X_test)[:, 1]
preds = (probs > best_t).astype(int)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
```

    Accuracy: 0.972
                  precision    recall  f1-score   support
    
               0       0.99      0.98      0.98      1847
               1       0.81      0.83      0.82       153
    
        accuracy                           0.97      2000
       macro avg       0.90      0.91      0.90      2000
    weighted avg       0.97      0.97      0.97      2000
    
    

We can note that recall has improved 30% with this adjustment. There is a tradeoff here however, as precision has dropped from 93% to 81%. This suggests our model is now hitting a higher number of false positives (93% to 81%). In the case of healthcare fraud, the 12% increase in false positive rate is more than acceptable for a 30% increase in detection rates. False positives flagged by the model can be weeded out in the review process. 

### Features
Let's look at which features were most important to our model in making its determination.


```python
importance = pd.Series(model.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False).head(10))
```

    Days_Between_Service_and_Claim           0.243330
    Claim_Amount                             0.099722
    Approved_Amount                          0.070125
    Number_of_Claims_Per_Provider_Monthly    0.045083
    Claim_Status_Rejected                    0.039321
    Claim_Status_Pending                     0.029277
    Patient_Age                              0.028903
    Prior_Visits_12m                         0.020479
    Procedure_Code                           0.019170
    Length_of_Stay                           0.017786
    dtype: float64
    

The most important feature is Days_Between_Service_and_Claim, with almost a quarter relevance.  Let's start by getting the averages between fraudlent and non-fraudlent claims.


```python
df.groupby("Is_Fraud")["Days_Between_Service_and_Claim"].mean()
```




    Is_Fraud
    0    15.447825
    1     2.974668
    Name: Days_Between_Service_and_Claim, dtype: float64



We can immediately see a huge disparity in average time between service and claim for legitimate and fraudulent claims. Legitimate claims average 15 days, whereas fraudlent claims average less than 3. Let's graph this out.


```python
sns.boxplot(
    x="Is_Fraud",
    y="Days_Between_Service_and_Claim",
    data=df
)

plt.title("Days Between Service and Claim by Fraud Status")
plt.show()
```


    
![png](output_21_0.png)
    


We can see that fraudlent claims were all resolved within a week. Although some legitimate claims were resolved within a week, no fraudulent claims were resolved outside a week. This suggests targeting claims with days between service and claim of one week or less would have caught all of the fraudlent cases in this dataset. Let's explore this heuristic further. To start, we will calculate how many legitimate cases had a service to claim period of one week or less.


```python
total_legit = df[df["Is_Fraud"] == 0].shape[0]

flagged_legit = df[
    (df["Is_Fraud"] == 0) &
    (df["Days_Between_Service_and_Claim"] <= 7)
].shape[0]

percent_flagged_legit = (flagged_legit / total_legit) * 100

print("Legit claims flagged:", flagged_legit)
print("Total legit claims:", total_legit)
print("Percent of legit claims flagged by heuristic:", percent_flagged_legit)
```

    Legit claims flagged: 1998
    Total legit claims: 9171
    Percent of legit claims flagged by heuristic: 21.786064769381746
    

We can now break down what the accuracy and precision of a heuristic like this would be.


```python
total_predictions = df[
    (df["Days_Between_Service_and_Claim"] <= 7)
].shape[0]

true_positives = df[
    (df["Is_Fraud"] == 1) &
    (df["Days_Between_Service_and_Claim"] <= 7)
].shape[0]

true_negatives = df[
    (df["Is_Fraud"] == 0) &
    (df["Days_Between_Service_and_Claim"] > 7)
].shape[0]

total_fraud = df[
    (df["Is_Fraud"] == 1)
].shape[0]

false_positives = total_predictions - true_positives
false_negatives = total_fraud - true_positives
accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall / (precision + recall))

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall", recall)
print("F1", f1)
```

    Accuracy: 0.8002
    Precision: 0.29324372125928544
    Recall 1.0
    F1 0.45350109409190364
    

This heuristic would catch all cases of fraud in the dataset but at a large cost to precision. Because there is a higher amount of legitimate cases than fraudulent ones, it results in a signifigant number of false positives. Due to this the F1 score drops well below the performance of our model (45% to 82%).

The second and third most important features in our model are Claim_Amount and Approved_Amount. Let's see if there are any insights we can gain from checking these averages.


```python
df.groupby("Is_Fraud")["Claim_Amount"].mean()

```




    Is_Fraud
    0    535.008352
    1    990.931797
    Name: Claim_Amount, dtype: float64




```python
df.groupby("Is_Fraud")["Approved_Amount"].mean()
```




    Is_Fraud
    0    469.154258
    1    545.871978
    Name: Approved_Amount, dtype: float64



Another major anomoly appears in the average claim amounts. Fraudulent claims average almost twice what legitimate claims do, a signifigant divergence. The divergence in approved amounts exists, suggesting larger approvals also share some correlation with fraud, however this deviation is smaller, perhaps due to adjustments before approval on large claim amounts.

### Conclusion

| Approach        | Recall (Fraud) | Precision | Legit Flag Rate |
|----------------|---------------|----------|-----------------|
| Heuristic (≤7d)| 100%          | 29%      | 21.8%           |
| ML Model       | 83%           | 81%      | ~3–5% (implied) |

By lowering the classification threshold from the default 0.5 to 0.29 we can significantly improve recall of our model (53% to 83%). This results in a reduction of precision (93% to 81%). We assume that the 12% increase in false positives is worth the 30% increase in recall as false positives will be caught in the claims review process.

The model identifies the most important feature as time between service and claim. Legitimate claims average more than two weeks between service and claim, while fraudlent claims tend to have a period shorter than 3 days. We can use this insight to draw some heuristics. Implementing an automatic review of claims whose period was less than one week would have flagged 100% of fraudlent claims. However due to a large increase in false positives, the impact on review workloads must be strongly considered, as 21% of all legimate claims would be flagged. 

Other important features are claim amounts and adjusted amounts. Fraudulent claim amounts average almost double what legitimate claims do (535 to 990). Approved amounts share this pattern, albiet weaker (469 to 545). The possibility of claims adjustment before approval likely normalizes claim amounts and weakens its potential as a heuristic.

For fraud detection moving forward, claims with a waiting period of less than a week for service should trigger automatic review. Claims with abnormally high claim amounts should also warrant additional review. 


