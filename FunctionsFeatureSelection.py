import pandas as pd
import numpy as np

from sklearn.inspection import permutation_importance
import shap
from sklearn.feature_selection import (
    SelectKBest, f_regression, RFECV, SequentialFeatureSelector, SelectFromModel
)
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def feature_selection(X, y, max_n_features=11):
    """Shows selected features according to different selection methods.
    
    Available selection methods are:
       - FStat : select number of most important variables according to the F statistic,
       - Ridge : select number of most important variables according to the t statistic from RidgeCV model,
       - RF : feature importance from the basic Random Forest model,
       - PermutatedRF : feature importance from the basic Random Forest model trained on 2-folds CV,
       - SHAP : .
    
    Parameters
    ----------
        X : features dataset,
        y : explainable variable,
        max_n_features : number of features to select.
        
    Returns
    ----------
        DataFrame with lists of columns selected in each method.
    """
    
    selector_knn = SelectKBest(f_regression, k=max_n_features)
    selector_knn.fit(X,y)

    selector_ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)
    importance = np.abs(selector_ridge.coef_)
    feature_names = np.array(X.columns)
    threshold = np.sort(importance)[-max_n_features-1] + 0.01
    sfm = SelectFromModel(selector_ridge, threshold=threshold).fit(X, y)
    
    selector_rf = RandomForestRegressor(random_state=0)
    selector_rf.fit(X, y)
    feature_names = X.columns
    importances = selector_rf.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    forest = RandomForestRegressor(random_state=0)
    forest.fit(X_train, y_train)
    
    selector_perm_imp = permutation_importance(forest, X_test, y_test)
    sorted_idx = selector_perm_imp.importances_mean.argsort()
    perm_importance = pd.DataFrame(dict(
        PermutatedRF=feature_names[sorted_idx],
        importance=selector_perm_imp.importances_mean[sorted_idx]
    )).sort_values(by='importance', ascending=False).head(max_n_features).reset_index(
    ).drop(columns='index').PermutatedRF.to_list()
    
    selector_shap = shap.TreeExplainer(forest)
    shap_values = selector_shap.shap_values(X_test)
    shap_sum = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame(dict(
        col = X.columns.tolist(),
        imp = shap_sum.tolist()
    )).sort_values('imp', ascending=False).head(max_n_features).col.to_list()
    
    imp_dict = pd.DataFrame(dict(
        FStat = selector_knn.get_feature_names_out().tolist(),
        Ridge = feature_names[sfm.get_support()],
        RF = forest_importances.sort_values(ascending=False).head(max_n_features).index.to_list(),
        PermutatedRF = perm_importance,
        SHAP = shap_importance
    ))
    return imp_dict

def show_model_ga_search_cv(model_grid, model, name, X, y,
                            problem_type='classification', cv=3, popsize=20, generations=30):
    """Shows best hyperparameters for a selected model through generic algoritm search.
       
    Parameters
    ----------
        model_grid : dictionary for predefined hyperparameters' ranges,
        model : selected model,
        name : name of the model,
        X : features dataset
        y : explainable variable,
        problem_type : 'classification' or 'regression', default='classification'.
                        In case of regression, RMSE metric will be used.
        cv : number of CV folds,
        popsize : initial population size (number of models),
        generations : number of generations (iterations).
        
    Returns
    ----------
        Dictionary of best hyperparameters,
        Models score.
    """
    
    if problem_type=='classification':
        scoring = 'accuracy'
    else:
        scoring = 'neg_root_mean_squared_error'
    
    model_grid_search_cv = GASearchCV(
        estimator=model,
        cv=cv,
        scoring=scoring,
        population_size=popsize,
        generations=generations,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.1,
        param_grid=model_grid,
        #criteria='max',
        algorithm='eaMuPlusLambda',
        n_jobs=-1,
        verbose=True,
        keep_top_k=4
    ).fit(X, y)
    print("\nModel:", name, "\n")
    print("Accuracy:", model_grid_search_cv.best_score_, "\n")
    print("Best params", model_grid_search_cv.best_params_, "\n")
    
def show_best_feature_set(features_original, features_centroids, features_selected, y, 
                          problem_type='classification'):
    model_grid_ga_rf = {
        'max_depth': Integer(10, 80),
        'max_features': Integer(1, 7),
        'min_samples_leaf': Integer(1, 7),
        'min_samples_split': Integer(2, 10),
        'n_estimators': Integer(25, 500)
    }
    
    print('Original:')
    show_model_ga_search_cv(
        model_grid_ga_rf, RandomForestRegressor(), 'random_forest', features_original, y,
        problem_type, generations=3
    )
    
    print('Centroids:')
    show_model_ga_search_cv(
        model_grid_ga_rf, RandomForestRegressor(), 'random_forest', features_centroids, y,
        problem_type, generations=3
    )
    
    print('Features selected:')
    show_model_ga_search_cv(
        model_grid_ga_rf, RandomForestRegressor(), 'random_forest', features_selected, y,
        problem_type, generations=3
    )
    
    #print('Expert-selected:')
    #print('FCA')
    #print('FCA selected')
    #print('Centroids selected')