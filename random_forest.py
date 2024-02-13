from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

print("Loading dataset:")
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

print("Vectorizing text data:")
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(newsgroups_train.data)
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

print("Defining parameter grid for Random Forest:")
parameter_grid = {
    'n_estimators': [50, 100, 200, 300, 500, 1000, 2500], 
    'max_depth': [None, 10, 20, 30], 
}

print("Setting up cross-validation:")
cv = StratifiedKFold(n_splits=5)

print("Starting Grid Search with cross-validation:")
grid_search = GridSearchCV(RandomForestClassifier(), parameter_grid, cv=cv, scoring='accuracy')
grid_search.fit(x_train_tfidf, newsgroups_train.target)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Printing out accuracies for all parameter combinations
print("\nAccuracies for all parameter combinations:")
for params, mean_score, scores in zip(
        grid_search.cv_results_['params'],  
        grid_search.cv_results_['mean_test_score'], 
        grid_search.cv_results_['std_test_score']):
    print(f"Parameters: {params}, Score: {mean_score:.3f} (+/- {scores:.3f})")

print("Predicting on test data:")
x_test_counts = count_vect.transform(newsgroups_test.data)
x_test_tfidf = tfidf_transformer.transform(x_test_counts)
best_estimator = grid_search.best_estimator_
predicted = best_estimator.predict(x_test_tfidf)

print("Calculating accuracy on test data:")
accuracy = accuracy_score(newsgroups_test.target, predicted)
print(f"Test Accuracy: {accuracy}")
