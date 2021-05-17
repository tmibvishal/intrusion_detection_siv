# 2018CS50426 Vishal Singh

import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def IP_octet_to_float_encode(ip_octets_string: str):
    f = 0.0
    for i, c in enumerate(ip_octets_string):
        v = ord(c)
        f = f * 1000 + float(v)
    return f


def IP_to_float_encode(IP_str: str):
    octets = IP_str.split(".")
    if len(octets) == 0:
        return 0.0
    f0 = IP_octet_to_float_encode(octets[0])
    if len(octets) == 1 or len(octets[1]) == 0:
        return f0
    f1 = IP_octet_to_float_encode(octets[1])
    f1 = f1 / (len(octets[1]) * 1000)
    return f0 + f1


def encode(data: pd.core.frame.DataFrame):
    ipcategory_name_dict = {'INTERNET': 1, 'PRIV-192': 2, 'PRIV-10': 3, 'PRIV-172': 4, 'PRIV-CGN': 5, 'LOOPBACK': 6,
                            'LINK-LOCAL': 7, 'BROADCAST': 8, 'MULTICAST': 9}

    category_name_dict = {'Attack': 1, 'Exploit': 2, 'Suspicious Reputation': 3, 'Control and Maintain': 4,
                          'Reconnaissance': 5, 'Malicious Activity': 6, 'Suspicious Network Activity': 7,
                          'Attack Preparation': 8, 'Compromise': 9, 'Suspicious Account Activity': 10,
                          'To Be Determined': 11}

    grandparent_category_dict = {"A": 1, "B": 2}

    string_columns = ["ipcategory_name", "categoryname", "grandparent_category"]
    numerical_columns = ["parent_category", "notified", "overallseverity", "timestamp_dist", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "score", "srcip_cd", "dstip_cd", "srcport_cd", "dstport_cd", "alerttype_cd", "direction_cd", "eventname_cd", "severity_cd", "reportingdevice_cd", "devicetype_cd", "devicevendor_cd", "domain_cd", "protocol_cd", "username_cd", "srcipcategory_cd", "dstipcategory_cd", "isiptrusted", "untrustscore", "flowscore", "trustscore", "enforcementscore",
                         "dstportcategory_dominate", "srcportcategory_dominate", "thrcnt_month", "thrcnt_week", "thrcnt_day", "p6", "p9", "p5m", "p5w", "p5d", "p8m", "p8w", "p8d"]

    columns_ignored = list(set(data.columns) - set(string_columns + numerical_columns))
    print(columns_ignored)

    # encoding the string columns
    data['ipcategory_name'] = data['ipcategory_name'].apply(lambda x: ipcategory_name_dict[x] if (x in ipcategory_name_dict) else 0)
    data['categoryname'] = data['categoryname'].apply(lambda x: category_name_dict[x] if (x in category_name_dict) else 0)
    data['grandparent_category'] = data['grandparent_category'].apply(lambda x: grandparent_category_dict[x] if (x in grandparent_category_dict) else 0)

    # encoding and adding ip address
    data['ip_prefix'] = data['ip'].apply(lambda x: ".".join(x.split('.')[:1]))
    data['ip_prefix'] = data['ip_prefix'].apply(lambda x: IP_octet_to_float_encode(x))
    # vc = data['ip_prefix'].value_counts()
    # data['ip_prefix'] = data['ip_prefix'].apply(lambda x: x if vc[x] > 50 else "other")
    data['ip_prefix_2'] = data['ip'].apply(lambda x: ".".join(x.split('.')[:2]))
    data['ip_prefix_2'] = data['ip_prefix_2'].apply(lambda x: IP_to_float_encode(x))

    data = data.drop(columns_ignored, axis=1)

    # fill all the NaN
    for column in data.columns:
        if data[column].isna().sum() > 0:
            data[column].fillna(0, inplace=True)

    # print(data.info())
    print(data.head())
    # print(data["ipcategory_name"].unique())
    # print(pd.get_dummies(data["categoryname"]))
    return data


def train_and_test(X_train, X_test, y_train, y_test, clf, model_name):
    print(f"------{model_name}------")
    begin = time.time()
    clf = clf.fit(X_train, y_train)
    print(f"Training took {time.time() - begin} sec")
    begin = time.time()
    predictions = clf.predict_proba(X_test)
    print(f"Prediction took {time.time() - begin} sec")
    # print(np.count_nonzero(y_test == predictions[:, 1])/len(y_test))
    print(f"roc_auc_score: {metrics.roc_auc_score(y_test, predictions[:, 1])}")
    print(f"accuracy_score: {metrics.accuracy_score(y_test, clf.predict(X_test))}")
    print(f"confusion_matrix\n {metrics.confusion_matrix(y_test, clf.predict(X_test))}")


training_data = pd.read_csv("/Users/vishal/Downloads/iitd_things/6th_Sem/siv810/Project/clion/data/cybersecurity_training/cybersecurity_training.csv", delimiter="|")
print(len(training_data.columns))
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(training_data.head())
training_data = encode(training_data)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(training_data.head())

# Separating into train (0.7) and test data (0.3)
y = training_data['notified']
X = training_data.drop('notified', axis=1)
test_size = 0.3
seed = random.randint(0, 1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# Logistic Regression
clf = LogisticRegression(max_iter=1000,  class_weight='balanced', multi_class='ovr') # One-vs-Rest and One-vs-One (each calssifier for each pair)
train_and_test(X_train, X_test, y_train, y_test, clf, "Logistic Regression")

# Gaussian Naive Bayes
clf = GaussianNB()
train_and_test(X_train, X_test, y_train, y_test, clf, "Gaussian Naive Bayes")

# Support Vector Machine
clf = SVC(probability=True) # kernel='rbf' default
train_and_test(X_train, X_test, y_train, y_test, clf, "Support Vector Machine")

# Neural Network
clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(100, 50, 10, 5), activation='logistic', max_iter=1000)
train_and_test(X_train, X_test, y_train, y_test, clf, "Neural Network")

# Decision Tree
clf = DecisionTreeClassifier()
train_and_test(X_train, X_test, y_train, y_test, clf, "Decision Tree")

# Random Forest
clf = RandomForestClassifier()
train_and_test(X_train, X_test, y_train, y_test, clf, "Random Forest")

# Final Method
clf = RandomForestClassifier(n_estimators=1000, max_depth=25, random_state=0)
train_and_test(X_train, X_test, y_train, y_test, clf, "Random Forest")

# Test Data
print("Test Data")
test_data = pd.read_csv("/Users/vishal/Downloads/iitd_things/6th_Sem/siv810/Project/clion/data/cybersecurity_test/cybersecurity_test.csv", delimiter="|")
test_data = encode(test_data)
begin = time.time()
clf = clf.fit(X, y)
print(f"Training took {time.time() - begin} sec")
begin = time.time()
predictions = clf.predict_proba(X_test)
print(f"Prediction took {time.time() - begin} sec")
print(np.count_nonzero(predictions[:, 1] > 0.5)/len(test_data))
print(np.count_nonzero(predictions[:, 1] > 0.5))
print(len(test_data))


# print(np.count_nonzero(y_test == predictions[:, 1])/len(y_test))
# print(f"roc_auc_score: {metrics.roc_auc_score(y_test, predictions[:, 1])}")
# print(f"accuracy_score: {metrics.accuracy_score(y_test, clf.predict(X_test))}")
# print(f"confusion_matrix\n {metrics.confusion_matrix(y_test, clf.predict(X_test))}")