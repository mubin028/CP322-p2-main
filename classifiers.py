from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score 

def fold(grid_search, x_train_tfidf, x_test_tfidf, train, test):
    grid_search.fit(x_train_tfidf, train.target)
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    for params, mean_score, scores in zip(
            grid_search.cv_results_['params'],  
            grid_search.cv_results_['mean_test_score'], 
            grid_search.cv_results_['std_test_score']):
        print(f"Parameters: {params}, Score: {mean_score:.3f} (+/- {scores:.3f})")
    best_estimator = grid_search.best_estimator_
    predicted = best_estimator.predict(x_test_tfidf)
    accuracy = accuracy_score(test.target, predicted)
    print(f"Test Accuracy: {accuracy:.4f}")
    return

def lr_fold(x_train_tfidf, x_test_tfidf, train, test, parameters):
    cv = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(LogisticRegression(), param_grid=parameters, cv=cv, scoring='accuracy')
    fold(grid_search, x_train_tfidf, x_test_tfidf, train, test)
    return 


def decision_tree_fold(x_train_tfidf, x_test_tfidf, train, test, parameters):
    cv = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=parameters, cv=cv, scoring='accuracy')
    fold(grid_search, x_train_tfidf, x_test_tfidf, train, test)
    return 


def svc_fold(x_train_tfidf, x_test_tfidf, train, test, parameters):
    cv = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(LinearSVC(dual='auto'), param_grid=parameters, cv=cv, scoring='accuracy')
    fold(grid_search, x_train_tfidf, x_test_tfidf, train, test)
    return 

def ada_boost_fold(x_train_tfidf, x_test_tfidf, train, test, parameters):
    cv = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(AdaBoostClassifier(), param_grid=parameters, cv=cv, scoring='accuracy')
    fold(grid_search, x_train_tfidf, x_test_tfidf, train, test)
    return 


def random_forest_fold(x_train_tfidf, x_test_tfidf, train, test, parameters):
    cv = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid=parameters, cv=cv, scoring='accuracy')
    fold(grid_search, x_train_tfidf, x_test_tfidf, train, test)
    return 
