# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:33:55 2023

@author: marci
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#%% MATPLOTLIB CONFIG
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
def get_custom_color_palette_hash():
    return LinearSegmentedColormap.from_list("", [
                                            '#1d3557', '#669bbc', '#7684df',
                                            '#9c9c9c',
                                            '#c3c3c3', '#cf354c', '#f392a2',
                                            '#fcbf49', "#eae2b7" ])
cmap = get_custom_color_palette_hash()

for lfont in fm.findSystemFonts(fontpaths=None, fontext="ttf"):
    if "Roboto" in lfont:
        print(lfont)
        fm.fontManager.addfont(lfont)

plt.rcParams['font.sans-serif'] = "Roboto"#'Montserrat'

def set_pub():
    plt.rcParams.update({
        "font.weight": "bold",  # bold fonts
        "lines.linewidth": 1,   # thick lines
        "lines.color": "k",     # black lines
        "grid.color": "0.5",    # gray gridlines
        "grid.linestyle": "-",  # solid gridlines
        "grid.linewidth": 0.5,  # thin gridlines
        "savefig.dpi": 300,     # higher resolution output.
    })
    
set_pub()
#%%
RANDOM_STATE = 42

os.chdir("C:\\Users\\marci\\dev\\FinalProject_IronHack\\dansmarue")
data_path = "data/dmr_historique"

path_output= "model_results"


#%% FUNCTIONS 
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets using sklearn's train_test_split function.
    
    Parameters:
    X (pandas DataFrame): The feature matrix
    y (pandas Series): The target variable
    test_size (float): The proportion of data to include in the test set
    random_state (int): The random seed to use for reproducibility
    
    Returns:
    X_train (pandas DataFrame): The feature matrix for the training set
    X_test (pandas DataFrame): The feature matrix for the testing set
    y_train (pandas Series): The target variable for the training set
    y_test (pandas Series): The target variable for the testing set
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test, scaler, to_frame = True):
    """
    Function used to scale data for the X train and test dataframes. 
    The scaler must be an empty scaled fintion

    Parameters
    ----------
    X_train : pd.DataFrame or numpy array
        DESCRIPTION.
    X_test : pd.DataFrame or numpy array
        DESCRIPTION.
    scaler : TYPE
        DESCRIPTION.

    Returns
    -------
    X_train_scaled : TYPE
        DESCRIPTION.
    X_test_scaled : TYPE
        DESCRIPTION.

    """
    og_index_train = X_train.index
    og_index_test = X_test.index
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
   
    if to_frame:
        X_train_scaled_df = pd.DataFrame(columns=X_train.columns, data = X_train_scaled,
                                         index=og_index_train)
        X_test_scaled_df = pd.DataFrame(columns=X_train.columns, data = X_test_scaled,
                                        index=og_index_test)
        return X_train_scaled_df, X_test_scaled_df
    else:
        return X_train_scaled, X_test_scaled

def downsampling(X,y, column_target="Bankrupt?"):
    """
    Downsampling strategy for balancing data. We recommend using this in the 
    splitted train and test set.

    Parameters
    ----------
    X : Dataframe or numpy array
        Features.
    y : Dataframe or numpy array
        Target variable values.
    column_target : TYPE, optional
        DESCRIPTION. The default is "Bankrupt?".

    Returns
    -------
    X_down : TYPE
        DESCRIPTION.
    y_down : TYPE
        DESCRIPTION.

    """
    data = pd.concat([X,y],axis=1)
    cat_down = y.value_counts().sort_values().index[0]
    category_1 = data[data[column_target] == cat_down] # positive class (minority)
    c1_len = y.value_counts().sort_values().iloc[0]
    data_down = category_1.copy()
    # downsample the majority class to the size of the positive class using pandas sample method
    for cat_ in y.unique():
        if cat_ != cat_down:
            category_0 = data[data[column_target] == cat_] # negative class (majority)
            if category_0.shape[0]/c1_len > 0.15:
                category_0_down = category_0.sample(c1_len)
                # reassemble the data
                data_down = pd.concat([data_down, category_0_down], axis=0)
                
            else:
                print("Only downsample if there is a gap of more than 15%")
                # reassemble the data
                data_down = pd.concat([data_down, category_0_down], axis=0)
    # shuffle the data
    data_down = data_down.sample(frac=1) # frac spe
    X_down = data_down[X.columns]
    y_down = data_down[column_target]
    return X_down, y_down

from imblearn.over_sampling import SMOTE
def upsamplingSMOTE(X,y, strategy=None):
    smote = SMOTE(strategy=strategy)
    X_sm, y_sm = smote.fit_resample(X, y)
    return X_sm, y_sm


from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.gridspec as gridspec
    
def calculate_metrics_error(y_test,y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_ = r2_score(y_test,y_pred)
    bias = np.mean((y_test-y_pred)/y_test)
    mape = np.mean(np.abs((y_test-y_pred)/y_test))*100
    return rmse, r2_, bias, mape
def check_model_results_class(model, X_train, y_train, X_test, y_test,return_model=False):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)


    performance_df = pd.DataFrame({'Error_metric': ['Accuracy','Precision','Recall'],
                                'Train': [accuracy_score(y_train, y_pred_train),
                                            precision_score(y_train, y_pred_train,
                                                            average='macro'),
                                            recall_score(y_train, y_pred_train,
                                                         average='macro')],
                                'Test': [accuracy_score(y_test, y_pred_test),
                                            precision_score(y_test, y_pred_test,
                                                            average='macro'),
                                            recall_score(y_test, y_pred_test,
                                                         average='macro')]})
    
    cell_text = []
    cell_text.append([f"{accuracy_score(y_train, y_pred_train):.2f}",f"{accuracy_score(y_test, y_pred_test):.2f}"])
    cell_text.append([f"{precision_score(y_train, y_pred_train,average='macro'):.2f}",
                      f"{precision_score(y_test, y_pred_test,average='macro'):.2f}"])
    cell_text.append([f"{recall_score(y_train, y_pred_train,average='macro'):.2f}",
                      f"{recall_score(y_test, y_pred_test,average='macro'):.2f}"])
    #display(performance_df)
    cm = confusion_matrix( y_train, y_pred_train)
    cm2 = confusion_matrix( y_test, y_pred_test)
    
    # Plot figure with subplots of different sizes
    fig = plt.figure(figsize=(9.5, 7), layout="constrained",dpi=192)
    spec = fig.add_gridspec(2, 2)
        
    ax00 = fig.add_subplot(spec[0, 0])
    
    ax01 = fig.add_subplot(spec[0, 1])
    
    ax1 = fig.add_subplot(spec[1, :])
    
    ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=["0","3","1","8", "9"]).plot(ax=ax00,colorbar=False)
    ConfusionMatrixDisplay(confusion_matrix=cm2,
                                display_labels=["0","3","1","8","9"]).plot(ax=ax01,colorbar=False)

   
    the_table = ax1.table(cellText=cell_text,
                          rowLabels=['Accuracy','Precision','Recall'],
                          colLabels=["Train","Test"],
                          loc='center')
    ax1.axis("off") 
    fig.tight_layout()
    if return_model:
        return performance_df, fig, y_pred_train,y_pred_test, model
    else:
        return performance_df,fig, y_pred_train,y_pred_test
    return performance_df,fig, y_pred_train,y_pred_test
def check_model_results(model, X_train, y_train, X_test, y_test, return_model=False):
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    
    performance_df_train = pd.DataFrame([calculate_metrics_error(y_train, y_pred_train)], columns= ['rmse','r2','bias','mape'])
    performance_df_test = pd.DataFrame([calculate_metrics_error(y_test, y_pred_test)], columns= ['rmse','r2','bias','mape'])
    
    textstr_train = '\n'.join((
    r'$RMSE=%.2f$' % (performance_df_train["rmse"].values[0], ),
    r'$r^2=%.2f$' % (performance_df_train["r2"].values[0], ),
    r'$bias=%.2f$' % (performance_df_train["bias"].values[0], ),
    r'$MAPE=%.2f$' % (performance_df_train["mape"].values[0],)))
    
    textstr_test= '\n'.join((
    r'$RMSE=%.2f$' % (performance_df_test["rmse"].values[0], ),
    r'$r^2=%.2f$' % (performance_df_test["r2"].values[0], ),
    r'$bias=%.2f$' % (performance_df_test["bias"].values[0], ),
    r'$MAPE=%.2f$' % (performance_df_test["mape"].values[0],)))
    
    maxy = [np.max(y_test),np.max(y_train),np.max(y_pred_train),np.max(y_pred_test)]
    fig, [ax1,ax2] = plt.subplots(1,2, figsize=(10,5), sharex=True, sharey=True,dpi=300)
    ax1.set_ylim(0,max(maxy))
    ax1.set_xlim(0,max(maxy))
    ax1.scatter(y_train, y_pred_train, color = "#1d3557",alpha=0.7, label="train set")
    # place a text box in upper left in axes coords
    ax1.text(0.05, 0.95, textstr_train, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax2.scatter(y_test, y_pred_test, color="#354ccf",alpha=0.7, label="test set")
    ax2.text(0.05, 0.95, textstr_test, transform=ax2.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax1.set_xlabel("ground thruth")
    ax2.set_xlabel("ground thruth")
    ax1.set_ylabel("prediction")
    ax2.set_ylabel("prediction")
    if return_model:
        return performance_df_train,performance_df_test, y_pred_train,y_pred_test, fig, ax1, ax2, model
    else:
        return performance_df_train,performance_df_test, y_pred_train,y_pred_test, fig, ax1, ax2
from sklearn.metrics import roc_auc_score

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):
    
    #creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict

#%% SELECTING FINAL FEATURES TO RUN THE MODEL
from sklearn.preprocessing import OneHotEncoder
data_og = pd.read_csv("model_prep/model_dataset_8_class.csv")
columns_clean = [ x.strip() for x in data_og.columns]
data_og.columns = columns_clean
data_og.rename(columns={'Loyers de référence':"rent_p_m2"},inplace=True)
target_column = 'category_EN_code'
test_name = "nineclass"
slice_dataset=True
#cutting in half to run model
if slice_dataset:
    slice_ = int(data_og.shape[0]/4)
    data_og =data_og.sample(slice_)

drop_features=True
if drop_features:
    features_to_drop= ["pop_2021"]
    final_df = data_og.drop(columns = features_to_drop)
    
    features = final_df.columns.to_list()
    features.pop(features.index(target_column))
else:
    final_df = data_og.copy()
    # final_df.drop(columns = ["date","date_input"],inplace=True)
    features = data_og.columns.to_list()
    features.pop(features.index("target"))
    features.pop(features.index("date"))
    features.pop(features.index("date_input"))
    features.pop(features.index("quartier_id"))
    # ['quartier_id', 'surface_km2', 'surface_parking_m2',
    #         'green_space_surf_m2', 'zti', 'density_pop_2021',
    #        'menages_km2', 'pop_2021', 'number_shops_p_km2',
    #        'number_amenities_p_km2', 'n_places_p_km2','etat_trafic_0',
    #        'etat_trafic_1', 'etat_trafic_2', 'etat_trafic_3', 'etat_trafic_4',
    #        'length_m_voie']
    
final_df.isnull().sum()
final_df.fillna(0,inplace=True)
#%%
print(f"Size of the dataframe {final_df.shape[0]}")
X = final_df.loc[:,features].copy()
y = final_df[target_column]

#Check balance of target variable
y.value_counts()/len(y)*100
#Downsampling
#%% OTHER CATEGORIES MAKE UP FOR 10% ONLY SO WE CAN TRANSFORM INTO "ANOTHER" CATEGORY AND RELABEL
cat_to_keep = [0,3,1,8]
y_to_relab = y[(y==2)|(y>3)&(y!=8)].index
y[y_to_relab] = 9
#%% SPLITTING DATA BEFORE TREATMENT

X_train, X_test, y_train, y_test = split_data(X,y, test_size = 0.25)

#%% SCALING FEATURES
import pickle
#ONLY NON ENCODED SO ANY FLOAT
numerical_cols = X.select_dtypes(include="float").columns
numerical_cols = numerical_cols.drop("lat")
numerical_cols = numerical_cols.drop("lon")
scaler_function = "MinMaxScaler"
to_frame = True
if scaler_function == "Standard":
    scaler = StandardScaler()
    X_train_scaled,X_test_scaled = scale_data(X_train.loc[:,numerical_cols], X_test.loc[:,numerical_cols], scaler,to_frame=to_frame)
elif  scaler_function == "PowerTrans":
    scaler = PowerTransformer()
    X_train_scaled,X_test_scaled = scale_data(X_train.loc[:,numerical_cols], X_test.loc[:,numerical_cols], scaler,to_frame=to_frame)
else:
    scaler = MinMaxScaler()
    X_train_scaled,X_test_scaled = scale_data(X_train.loc[:,numerical_cols],X_test.loc[:,numerical_cols], scaler,to_frame=to_frame)
with open(f"model_prep/scaler_{test_name}.pkl","wb") as f:
    pickle.dump(scaler,f)
#%%    
X_not_float = X.select_dtypes(exclude="float").columns
if len(X_not_float)>1:
    X_train_scaled = pd.concat([X_train_scaled, X_train.loc[:,X_not_float]],axis=1)
    X_train_scaled = pd.concat([X_train_scaled, X_train.loc[:,["lat","lon"]]],axis=1)
    X_test_scaled = pd.concat([X_test_scaled, X_test.loc[:,X_not_float]],axis=1)
    X_test_scaled = pd.concat([X_test_scaled,X_test.loc[:,["lat","lon"]]],axis=1)

if to_frame:
    X_train_scaled.reset_index(inplace=True, drop=True)
    X_test_scaled.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)
#%%BECAUSE OF IMBALANCE WE FOCUSED ON THE FIRST 4 CATEGORIES AND "OTHER"
# target_class= [0,1,2,4]
# subset_class = True
# if subset_class:

#%% IF IMBALANCED DATA WE CAN USE DOWNSAMPLING OR SMOTE

balancing = "down"
if balancing == "down":
    X_train_balanced, y_train_balanced = downsampling(X_train_scaled, y_train,column_target=target_column)
elif balancing =="smote":
    X_train_balanced, y_train_balanced = upsamplingSMOTE(X_train_scaled, y_train)
print("Balanced only training samples")
# --------------
# # transform the dataset
# strategy = {0:100, 1:100, 2:200, 3:200, 4:200, 5:200}
# oversample = SMOTE(sampling_strategy=strategy)
# X, y = oversample.fit_resample(X, y)
# # summarize distribution
# counter = Counter(y)
# for k,v in counter.items():
#  per = v / len(y) * 100
#  print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# # plot the distribution
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()
#%% BUILDING AND TESTING DIFFERENT MODELS
# Saving figures automatically in path_images
path_images = "C:\\Users\marci\dev\FinalProject_IronHack\dansmarue\model_results"
#%%FIRST TEST KNN
test_knn=False
if test_knn:
    #Setup arrays to store training and test accuracies
    neighbors = np.arange(8,20,50)
    train_accuracy =np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    
    for i,k in enumerate(neighbors):
        #Setup a knn classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k)
        
        #Fit the model
        if balancing:
            knn.fit(X_train_balanced, y_train_balanced)
            #Compute accuracy on the training set
            train_accuracy[i] = knn.score(X_train_balanced, y_train_balanced)
            
            #Compute accuracy on the test set
            test_accuracy[i] = knn.score(X_test_scaled, y_test) 
        else:
            knn.fit(X_train_scaled, y_train)
            #Compute accuracy on the training set
            train_accuracy[i] = knn.score(X_train_scaled, y_train)
            
            #Compute accuracy on the test set
            test_accuracy[i] = knn.score(X_test_scaled, y_test) 
        
        
    #Generate plot
    plt.title('k-NN Varying number of neighbors')
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training accuracy')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.show()
    


#%%
from sklearn.linear_model import LogisticRegression
import time
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
perf_models = pd.DataFrame()
base_model = False
if base_model:
    model_1 = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    model_2 = LogisticRegression(multi_class="multinomial",random_state=RANDOM_STATE)
    model_3 = DecisionTreeClassifier(random_state=RANDOM_STATE)
    
    dic_models = { "RandomForestClassifier": model_1,
                  "LogisticRegression":model_2,
                  "DecisionTreeClassifier":model_3}
    
    list_roc = []
    for model_name, model_ in dic_models.items():
        start_time = time.time()
        if balancing:
            performance_df,fig, y_pred_train,y_pred_test = check_model_results_class(model_, X_train_balanced, y_train_balanced, X_test_scaled, y_test)
           
        else:
           performance_df,fig, y_pred_train,y_pred_test = check_model_results_class(model_, X_train_scaled, y_train, X_test_scaled, y_test)
        end_time = time.time()
        print(f"{end_time -start_time }")
        
        fig.suptitle(model_name)
        fig.tight_layout()
        fig.savefig(path_images + f"\{model_name}_testv3_{test_name}_PT.png",dpi=192)
        
        print("ROC_AUC")
        roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred_test)
        roc_auc_dict
        list_roc.append(roc_auc_dict)
    df_roc = pd.DataFrame(list_roc)
    df_roc.to_csv(f"{path_images}\\three_basic_testv3_{test_name}_PT.csv")
    perf_models.to_csv(f"{path_images}\\three_basic_testv3_{test_name}_PT.csv",index=False)
#%%
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
model_name = "stacking"
stacking = False
if stacking :
    estimators = [('rf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
                               ('svr', make_pipeline(StandardScaler(), #('svr', make_pipeline(StandardScaler(),
                       LinearSVC(random_state=RANDOM_STATE)))]
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    
    performance_df,fig, y_pred_train,y_pred_test = check_model_results_class(model_, X_train, y_train, X_test, y_test)
    
    fig.suptitle(model_name)
    fig.tight_layout()
    fig.savefig(path_images + f"\{model_name}_testClass_{test_name}_PT.png",dpi=200)
    perf_models.to_csv(f"{path_images}\stacking_basicClass_test_{test_name}_PT.csv",index=False)

#%%Building GridSearchV For GaussianNB
import json
grid_search =True
if grid_search:
    from sklearn.model_selection import GridSearchCV, cross_val_score
    model_name = "RandomForestClassifier"
    param_grid = { 
        'n_estimators': [600, 500, 400],
        'max_depth' : [8,10],
        'class_weight':['balanced']}

    clf = RandomForestClassifier( random_state=RANDOM_STATE)
    
    if balancing:
        grid_search = GridSearchCV(clf, param_grid, cv=5,return_train_score=True,n_jobs=-1)
        grid_search.fit(X_train_balanced, y_train_balanced)
        best_params = grid_search.best_params_ #To check the best set of parameters returned
        print(best_params)
        with open(f"{path_images}\\{model_name}CV_bestparams_RFclass.csv", "w") as f:
            json.dump(best_params,f)
        #USING BEST PARAMETERS FOUND TO GET BEST SCORES AFTER CROSS VALIDATION
        
        clf = RandomForestClassifier(**best_params)
        cross_val_scores = cross_val_score(clf, X_train_balanced, y_train_balanced, cv=5)
        print(np.mean(cross_val_scores))
        #USING BEST PARAMETERS FOUND
    
        model_ = RandomForestClassifier( **best_params)
        performance_df,fig, y_pred_train,y_pred_test ,best_model_  = check_model_results_class(model_, 
                                                                           X_train_balanced, 
                                                                           y_train_balanced, X_test_scaled, y_test,return_model=True)
    else:
        grid_search = GridSearchCV(clf, param_grid, cv=5,return_train_score=True,n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        best_params = grid_search.best_params_ #To check the best set of parameters returned
        print(best_params)
        with open(f"{path_images}\\{model_name}CV_bestparams_RFclass.csv", "w") as f:
            json.dump(best_params,f)
        #USING BEST PARAMETERS FOUND TO GET BEST SCORES AFTER CROSS VALIDATION
        
        clf = RandomForestClassifier(**best_params)
        cross_val_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
        print(np.mean(cross_val_scores))
    
        #USING BEST PARAMETERS FOUND
    
        model_ = RandomForestClassifier( **best_params)
        performance_df,fig, y_pred_train,y_pred_test ,best_model_  = check_model_results_class(model_, 
                                                                           X_train_scaled, 
                                                                           y_train, X_test_scaled, y_test,return_model=True)
        
    with open(f"{path_output}\{model_name}CV_testclass_{test_name}.pkl","wb") as f:
        pickle.dump(best_model_,f)
        
    fig.suptitle(model_name + "_CV")
    fig.tight_layout()
    fig.savefig(path_images + f"\{model_name}CV_testclass_{test_name}.png",dpi=200)
    perf_models.to_csv(f"{path_images}\{model_name}CV_testclass_{test_name}.csv",index=False)
    
    
    #FOR VISUALIZATION
    #transform y_pred_test
    
    df_plot_= pd.concat([X_test_scaled, y_test],axis=1)
    df_plot_["y_pred"] =  y_pred_test
    df_plot_.to_csv(f"{path_output}\{model_name}CV_testclass_{test_name}_ypredtest.csv",index=False)
    
