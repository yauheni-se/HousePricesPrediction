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
        KNN = selector_knn.get_feature_names_out().tolist(),
        Ridge = feature_names[sfm.get_support()],
        RF = forest_importances.sort_values(ascending=False).head(max_n_features).index.to_list(),
        PermutatedRF = perm_importance,
        SHAP = shap_importance
    ))
    return imp_dict

def show_model_ga_search_cv(model_grid, classifier, name, X, y, cv=3, popsize=20, generations=30):
    model_grid_search_cv = GASearchCV(
        estimator=classifier,
        cv=cv,
        #scoring='accuracy',
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