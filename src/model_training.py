from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_model_configs(random_state: int = 42):
    """Return models and their hyperparameter grids."""
    return {
        "Logistic Regression": (
            LogisticRegression(
                random_state=random_state,
                class_weight="balanced",
                solver="liblinear",
                max_iter=1000,
            ),
            {
                "model__C": [0.1, 1, 10],
                "model__penalty": ["l1", "l2"],
            },
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [3, 5, 7],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "Support Vector Machine": (
            SVC(
                random_state=random_state,
                class_weight="balanced",
                probability=True,
            ),
            {
                "model__C": [0.1, 1, 10],
                "model__gamma": ["scale", "auto"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=random_state),
            {
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5, 10],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 10, 20],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=random_state),
            {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.05, 0.1, 0.5],
            },
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=random_state),
            {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.05, 0.1, 0.5],
            },
        ),
        "Naive Bayes": (
            GaussianNB(),
            {},
        ),
    }


def tune_model(preprocessor, model, param_grid, X_train, y_train, cv=3):
    """Create a preprocessing/model pipeline and tune it."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    if not param_grid:
        pipeline.fit(X_train, y_train)
        return pipeline, None

    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_
