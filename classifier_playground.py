from sklearn.datasets import make_moons, make_blobs, make_circles, make_classification, make_multilabel_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, f1_score, recall_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import inspect
import streamlit as st
import pandas as pd

st.title("Choose your classifier")

# 1. select dataset
def select_dataset(dataset_name, noise=0, size=100, cluster_std=1):
    if dataset_name == 'moons':
        x, y = make_moons(n_samples=size, noise=noise)
        
    elif dataset_name == 'blobs':
        x, y = make_blobs(n_samples=size, centers=2, cluster_std=cluster_std)
    elif dataset_name == 'circles':
        x, y = make_circles(n_samples=size, noise=noise)
    elif dataset_name == 'linear':
        x, y = make_classification(n_samples=size, n_features=2, n_redundant=0, n_informative=2,
                                   n_clusters_per_class=1, class_sep=noise)
    
    return x, y

##### 
# There should be four functions for models
# 
# 1. select model
# 2. change model params
# 3. update model params
# 4. change model
#  
#####

### model section

# 1. select model
def select_model(model_name, x, y, model_params, test_size=0.20):

    if model_name == 'SVC':
        model = model_dict.get(model_name)(probability=True)
    elif model_name == 'SGDClassifier':
        model = model_dict.get(model_name)(loss='modified_huber')
    else:
        model = model_dict.get(model_name)()

    st.markdown("Read the docs")
    st.markdown(pred + model_dict_links.get(model_name))
    return model

def plot_result(model, x, y, test_size=0.20):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
    model.fit(x_train,y_train)
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    y_pred = model.predict(x_test)
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
    st.pyplot()
    df_cf_matrix = pd.DataFrame(confusion_matrix(y_pred, y_test), columns=["True", "False"], index=["Positive", "Negative"])
    st.dataframe(df_cf_matrix)
    df_metrics = pd.DataFrame({
        "F1 Score"          :f1_score(y_pred, y_test),
        "AUC Score"         :roc_auc_score(y_pred, y_test),
        "Accuracy Score"    :accuracy_score(y_pred, y_test),
        "Recall Score"      :recall_score(y_pred, y_test)
    },index=["Metrics"]).T
    print(df_metrics)
    st.dataframe(df_metrics)
    fpr, tpr, _ = roc_curve(y_pred, y_test)
    auc = roc_auc_score(y_pred, y_test)
    plt.plot(fpr,tpr,label="data 1, auc="+str('%.3f' % auc))
    plt.legend(loc=4)
    st.pyplot()
# 2. change model params

def choose_params(model_selected, model_dict):
    if model_selected == 'SVC':
        model = model_dict.get(model_selected)(probability=True)
    elif model_selected == 'SGDClassifier':
        model = model_dict.get(model_selected)(loss='modified_huber')
    else:
        model = model_dict.get(model_selected)()
    
    model_params = model.get_params()
    
    for p in model_params:
        if p == 'probability' and p == 'loss':
            pass
        elif type(model_params.get(p)) is int:
            st.sidebar.number_input(p, value=model_params.get(p))
        elif type(model_params.get(p)) is float:
            st.sidebar.number_input(p, value=model_params.get(p))
        elif type(model_params.get(p)) is bool:
            st.sidebar.selectbox(p, [True, False])
        

    return model, model_params


### This part takes too much time, because I`m an idiot
model_list_str = [
    'KNeighborsClassifier',
    'LogisticRegression',
    'SGDClassifier',
    'LinearDiscriminantAnalysis',
    'QuadraticDiscriminantAnalysis',
    'SVC'
]

pred = "https://scikit-learn.org/stable/modules/generated/sklearn."

model_links = [
    "neighbors.KNeighborsClassifier.html",
    "linear_model.LogisticRegression.html",
    "linear_model.SGDClassifier.html",
    "discriminant_analysis.LinearDiscriminantAnalysis.html",
    "discriminant_analysis.QuadraticDiscriminantAnalysis.html",
    "svm.SVC.html"
]
model_list = [
    KNeighborsClassifier,
    LogisticRegression,
    SGDClassifier,
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    SVC
]

model_dict = dict(zip(model_list_str, model_list))
model_dict_links = dict(zip(model_list_str, model_links))
###
h = .02 # gridsize
cm = plt.cm.RdBu # colormap
cm_bright = ListedColormap(['#FF0000', '#0000FF']) # cm params


dataset_name = st.selectbox('Select your dataset', ['moons', 'blobs', 'circles', 'linear']) # select your dataset

st.sidebar.title("Select the dataset parameters") # seleect your dataset params
if dataset_name == 'blobs':
    cluster_std = st.sidebar.number_input('Cluster std', value=1.) # diferent param name for this dataset type
else:
    noise = st.sidebar.number_input('Dataset noise', value=0.) # same param name for all others

size  = st.sidebar.selectbox('Dataset size', np.arange(100, 800, 100)) # data points

### selecting datasets (should be a function)
if dataset_name == 'blobs':
    dataset_params = {
        'size' :size,
        'std'  :cluster_std}
else:
    dataset_params = {
        'size' :size,
        'noise':noise}

if dataset_name == 'blobs':
    size, std = dataset_params.values()
    x, y = select_dataset(dataset_name, size=size, cluster_std=std)
    
else:
    size, noise = dataset_params.values()
    x, y = select_dataset(dataset_name, noise=noise, size=size)

###


### selecting the model
model_name = st.selectbox('Select your model', model_list_str)
###
### start the training with the model params!
play = st.button("Click here to train the model")

st.sidebar.title("Select the Model parameters")
model, model_params = choose_params(model_name, model_dict)
m = select_model(model_name, x, y, model_params)
if play:
    plot_result(m, x, y)
### this is the end

### TODO
# auto fitting parameters with gridsearchCV
# xgboost
# export your code