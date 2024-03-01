from matplotlib import pyplot as plt, ticker
import numpy as np 
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def preprocess():
    #Load in initial data / create DataFrames
    
    cand_comm_contrib = pd.DataFrame(pd.read_csv("./data/Contributions.csv"))
    candidates = pd.DataFrame(pd.read_csv("./data/Candidate master.csv"))
    candidates = candidates[["CAND_ID", "CAND_PTY_AFFILIATION", "CAND_ST"]]

    print(cand_comm_contrib.describe())

    data = pd.merge(cand_comm_contrib, candidates, on="CAND_ID")
    data.to_csv("data.csv",index=False)
    
def analysis():
    data = pd.DataFrame(pd.read_csv("./data.csv"))

    groups = data.groupby("CAND_PTY_AFFILIATION")["TRANSACTION_AMT"].sum().reset_index()
    groups = groups.sort_values(by="TRANSACTION_AMT", ascending=False).loc[groups["TRANSACTION_AMT"] > 10000]

    plt.figure(figsize=(10, 6))  
    bars = plt.bar(groups["CAND_PTY_AFFILIATION"], groups["TRANSACTION_AMT"], color='skyblue')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'${height:,.0f}', ha='center', va='bottom', fontsize=8)

    plt.xlabel('Party Affiliation')
    plt.ylabel('Total Contributions (USD)')
    plt.title('Total Committee Contributions by Party Affiliation')

    formatter = ticker.StrMethodFormatter('${x:,.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

data = pd.read_csv("./data.csv",low_memory=False)
label_encoder = LabelEncoder()

features = pd.DataFrame()
features["CITY"] = label_encoder.fit_transform(data["CITY"])
features["CMTE_ID"] = label_encoder.fit_transform(data["CMTE_ID"])
features['TRANSACTION_AMT'] = data["TRANSACTION_AMT"]
features["TRANSACTION_PGI"] = label_encoder.fit_transform(data["TRANSACTION_PGI"])
features["TRANSACTION_TP"] = label_encoder.fit_transform(data["TRANSACTION_TP"])
features["ENTITY_TP"] = label_encoder.fit_transform(data["ENTITY_TP"])

labels = np.reshape(label_encoder.fit_transform(data["CAND_PTY_AFFILIATION"]), (-1,1))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

def rf_model():
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42) 
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_test_1d = y_test.ravel()
    
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_test_1d, y_pred, average='weighted')
    recall = recall_score(y_test_1d, y_pred, average='weighted')
    f1 = f1_score(y_test_1d, y_pred, average='weighted')
    
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    feature_importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': rf_model.feature_importances_})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df)

def decision_tree_model():
    # Initialize the Decision Tree Classifier
    dt_model = DecisionTreeClassifier(random_state=42)

    # Train the model
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    y_test_1d = y_test.ravel()
    
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_test_1d, y_pred, average='weighted')
    recall = recall_score(y_test_1d, y_pred, average='weighted')
    f1 = f1_score(y_test_1d, y_pred, average='weighted')
    
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    feature_importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': dt_model.feature_importances_})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df)


def NaiveBayes_model():
    # Initialize Naive Bayes model
    nb_model = GaussianNB()

    # Train the Naive Bayes model
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    y_test_1d = y_test.ravel()
    
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_test_1d, y_pred, average='weighted')
    recall = recall_score(y_test_1d, y_pred, average='weighted')
    f1 = f1_score(y_test_1d, y_pred, average='weighted')
    
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

  
#preprocess()
#analysis()
rf_model()
NaiveBayes_model()
decision_tree_model()

    

