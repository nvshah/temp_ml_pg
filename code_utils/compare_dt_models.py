import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

def compare_decision_trees(X_train, X_test, y_train, y_test):
    """
    Compare different decision tree configurations with various optimization techniques.
    """
    results = {}
    
    # 1. Basic Decision Tree
    basic_dt = DecisionTreeClassifier(random_state=42)
    basic_dt.fit(X_train, y_train)
    results['basic'] = {
        'model': basic_dt,
        'train_score': basic_dt.score(X_train, y_train),
        'test_score': basic_dt.score(X_test, y_test)
    }
    
    # 2. Balanced Decision Tree
    balanced_dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    balanced_dt.fit(X_train, y_train)
    results['balanced'] = {
        'model': balanced_dt,
        'train_score': balanced_dt.score(X_train, y_train),
        'test_score': balanced_dt.score(X_test, y_test)
    }
    
    # 3. Pre-pruning with GridSearchCV
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'max_leaf_nodes': [None, 10, 20, 30]
    }
    
    grid_dt = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_dt.fit(X_train, y_train)
    results['prepruning'] = {
        'model': grid_dt.best_estimator_,
        'best_params': grid_dt.best_params_,
        'train_score': grid_dt.score(X_train, y_train),
        'test_score': grid_dt.score(X_test, y_test)
    }
    
    # 4. Post-pruning with cost complexity pruning
    postprune_dt = DecisionTreeClassifier(random_state=42)
    path = postprune_dt.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas
    
    # Test different alpha values
    alpha_results = []
    for alpha in alphas:
        dt = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
        dt.fit(X_train, y_train)
        train_score = dt.score(X_train, y_train)
        test_score = dt.score(X_test, y_test)
        alpha_results.append([alpha, train_score, test_score])
    
    # Find best alpha
    best_alpha = alphas[np.argmax([x[2] for x in alpha_results])]
    final_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
    final_dt.fit(X_train, y_train)
    
    results['postpruning'] = {
        'model': final_dt,
        'best_alpha': best_alpha,
        'train_score': final_dt.score(X_train, y_train),
        'test_score': final_dt.score(X_test, y_test)
    }
    
    return results

def print_model_comparison(results):
    """
    Print comparison of model performances
    """
    print("Model Comparison Results:")
    print("-" * 50)
    for model_name, result in results.items():
        print(f"\n{model_name.upper()} MODEL:")
        print(f"Training Score: {result['train_score']:.4f}")
        print(f"Testing Score: {result['test_score']:.4f}")
        
        if model_name == 'prepruning':
            print("Best Parameters:", result['best_params'])
        elif model_name == 'postpruning':
            print(f"Best Alpha: {result['best_alpha']:.6f}")