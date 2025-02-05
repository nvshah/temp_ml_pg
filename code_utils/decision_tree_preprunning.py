import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


def implement_prepruning(X_train, X_test, y_train, y_test):
    """
    Implement pre-pruning on decision tree using GridSearchCV
    Returns the best model and results visualization
    """
    # Define the parameter grid
    param_grid = {
        "max_depth": [3, 5, 7, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4, 6, 8],
        "max_leaf_nodes": [None, 10, 20, 30, 40, 50],
    }

    # Create base model
    dt = DecisionTreeClassifier(random_state=42)

    # Implement GridSearchCV
    grid_search = GridSearchCV(
        estimator=dt,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Get all results
    cv_results = pd.DataFrame(grid_search.cv_results_)

    # Calculate feature importance
    feature_importance = pd.DataFrame(
        {
            "feature": (
                X_train.columns
                if hasattr(X_train, "columns")
                else [f"Feature_{i}" for i in range(X_train.shape[1])]
            ),
            "importance": best_model.feature_importances_,
        }
    )
    feature_importance = feature_importance.sort_values("importance", ascending=False)

    return {
        "best_model": best_model,
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "train_score": best_model.score(X_train, y_train),
        "test_score": best_model.score(X_test, y_test),
        "cv_results": cv_results,
        "feature_importance": feature_importance,
    }


def plot_results(results):
    """
    Plot the results of pre-pruning analysis
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot feature importance
    sns.barplot(
        data=results["feature_importance"].head(10), x="importance", y="feature", ax=ax1
    )
    ax1.set_title("Top 10 Feature Importance")
    ax1.set_xlabel("Importance")
    ax1.set_ylabel("Features")

    # Plot training vs validation scores
    cv_results = results["cv_results"]
    ax2.scatter(
        cv_results["mean_test_score"], cv_results["mean_train_score"], alpha=0.5
    )
    ax2.plot([0, 1], [0, 1], "--k")
    ax2.set_xlabel("Validation Score")
    ax2.set_ylabel("Training Score")
    ax2.set_title("Training vs Validation Scores")

    plt.tight_layout()
    return fig


# Example usage:
"""
# Assuming you have X_train, X_test, y_train, y_test ready:
results = implement_prepruning(X_train, X_test, y_train, y_test)

print("Best Parameters:", results['best_params'])
print("\nBest Cross-validation Score:", results['best_score'])
print("Training Score:", results['train_score'])
print("Testing Score:", results['test_score'])

# Plot the results
plot_results(results)
plt.show()

# Print top 5 most important features
print("\nTop 5 Most Important Features:")
print(results['feature_importance'].head())
"""
