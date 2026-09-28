#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
#-----------------------------------------------------------------------------
#Load the dataset
df = pd.read_csv("mushrooms.csv")

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

# Understand the dataset
print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nTarget distribution:")
print(df["class"].value_counts())

#Check unique values
for column in df.columns:
    print(f"\n{column}:")
    print(df[column].unique())

#-----------------------------------------------------------------------------
#Clean the dataset
df = df.replace("?", np.nan)

print("\nMissing values after replacing '?':")
print(df.isnull().sum())

#Missing value imputation
for column in df.columns:
    if column != "class":
        df[column] = df[column].fillna(df[column].mode()[0])
print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())  

#Remove duplicate rows
print("Duplicates before:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicates after:", df.duplicated().sum())

print("New dataset shape:", df.shape)

#-----------------------------------------------------------------------------
#Training model

X = df.drop("class", axis=1)
y = df["class"]
print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

categorical_columns = X.columns.tolist()

print("Number of categorical features:",
      len(categorical_columns))  

#-----------------------------------------------------------------------------
#encoding categorical features and building the KNN model
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_columns
        )
    ]
)

metric="euclidean"
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean"
)
knn_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("knn", knn)
])

knn_pipeline.fit(X_train, y_train)

y_pred = knn_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy for K=5:", accuracy)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Edible", "Poisonous"],
    yticklabels=["Edible", "Poisonous"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix - K=5")

plt.show()
print(classification_report(y_test, y_pred))

#-----------------------------------------------------------------------------
#accuracy for different values of K
k_values = range(1, 31)

accuracies = []

for k in k_values:

    model = Pipeline([
        ("preprocessing", preprocessor),
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=k,
                metric="euclidean"
            )
        )
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    accuracies.append(accuracy)

results = pd.DataFrame({
    "K": list(k_values),
    "Accuracy": accuracies
})

print(results)
plt.figure(figsize=(10, 6))

plt.plot(
    results["K"],
    results["Accuracy"],
    marker="o"
)

plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K")
plt.xticks(k_values)
plt.grid(True)

plt.show()
best_test_row = results.loc[
    results["Accuracy"].idxmax()
]

print("Best K based on test accuracy:")
print(best_test_row)

#-----------------------------------------------------------------------------
#Cross-validation for different values of K

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for k in range(1, 31):

    model = Pipeline([
        ("preprocessing", preprocessor),
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=k,
                metric="euclidean"
            )
        )
    ])

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    cv_results.append({
        "K": k,
        "CV_Mean_Accuracy": scores.mean(),
        "CV_Std": scores.std()
    })

cv_results = pd.DataFrame(cv_results)

print(cv_results)

best_cv_row = cv_results.loc[
    cv_results["CV_Mean_Accuracy"].idxmax()
]

best_k_cv = int(best_cv_row["K"])

print("\nBest K using Cross Validation:", best_k_cv)

print(
    "Cross-validation accuracy:",
    best_cv_row["CV_Mean_Accuracy"]
)

plt.figure(figsize=(10, 6))

plt.errorbar(
    cv_results["K"],
    cv_results["CV_Mean_Accuracy"],
    yerr=cv_results["CV_Std"],
    marker="o",
    capsize=3
)

plt.xlabel("K")
plt.ylabel("Cross-Validation Accuracy")
plt.title("5-Fold Cross-Validation Accuracy vs K")
plt.xticks(range(1, 31))
plt.grid(True)

plt.show()

cv_best_model = Pipeline([
    ("preprocessing", preprocessor),
    (
        "knn",
        KNeighborsClassifier(
            n_neighbors=best_k_cv,
            metric="euclidean"
        )
    )
])

cv_best_model.fit(X_train, y_train)

y_pred_cv = cv_best_model.predict(X_test)

cv_test_accuracy = accuracy_score(
    y_test,
    y_pred_cv
)

print(
    "\nTest Accuracy using CV-selected K:",
    cv_test_accuracy
)

#-----------------------------------------------------------------------------
# odd values of K for cross-validation

odd_k_values = list(range(1, 30, 2))

odd_results = []

for k in odd_k_values:

    model = Pipeline([
        ("preprocessing", preprocessor),
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=k,
                metric="euclidean"
            )
        )
    ])

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    odd_results.append({
        "K": k,
        "CV_Accuracy": scores.mean()
    })

odd_results = pd.DataFrame(odd_results)

print(odd_results)
best_odd_row = odd_results.loc[
    odd_results["CV_Accuracy"].idxmax()
]

best_k_odd = int(best_odd_row["K"])

print("\nBest Odd K:", best_k_odd)

print(
    "CV Accuracy:",
    best_odd_row["CV_Accuracy"]
)

odd_best_model = Pipeline([
    ("preprocessing", preprocessor),
    (
        "knn",
        KNeighborsClassifier(
            n_neighbors=best_k_odd,
            metric="euclidean"
        )
    )
])

odd_best_model.fit(X_train, y_train)

y_pred_odd = odd_best_model.predict(X_test)

odd_test_accuracy = accuracy_score(
    y_test,
    y_pred_odd
)

print(
    "\nTest Accuracy using Odd K:",
    odd_test_accuracy
)

#-----------------------------------------------------------------------------
#Elbow method to find optimal K
plt.figure(figsize=(10, 6))

plt.plot(
    cv_results["K"],
    cv_results["CV_Mean_Accuracy"],
    marker="o"
)

plt.xlabel("K")
plt.ylabel("Cross-Validation Accuracy")
plt.title("Elbow-Style K Selection for KNN")
plt.xticks(range(1, 31))
plt.grid(True)

plt.show()

from sklearn.preprocessing import MinMaxScaler

x = cv_results["K"].values
y = cv_results["CV_Mean_Accuracy"].values

# Normalize x and y
x_norm = (x - x.min()) / (x.max() - x.min())
y_norm = (y - y.min()) / (y.max() - y.min())

# Line between first and last points
x1, y1 = x_norm[0], y_norm[0]
x2, y2 = x_norm[-1], y_norm[-1]

distances = np.abs(
    (y2 - y1) * x_norm
    - (x2 - x1) * y_norm
    + x2 * y1
    - y2 * x1
) / np.sqrt(
    (y2 - y1)**2 + (x2 - x1)**2
)

elbow_index = np.argmax(distances)

best_k_elbow = int(x[elbow_index])

print("Elbow-selected K:", best_k_elbow)

elbow_model = Pipeline([
    ("preprocessing", preprocessor),
    (
        "knn",
        KNeighborsClassifier(
            n_neighbors=best_k_elbow,
            metric="euclidean"
        )
    )
])

elbow_model.fit(X_train, y_train)

y_pred_elbow = elbow_model.predict(X_test)

elbow_test_accuracy = accuracy_score(
    y_test,
    y_pred_elbow
)

print(
    "Test Accuracy using Elbow-selected K:",
    elbow_test_accuracy
)

#-----------------------------------------------------------------------------
#comparison of different K selection methods
comparison = pd.DataFrame({
    "Method": [
        "Cross Validation",
        "Elbow Method",
        "Odd K"
    ],

    "Selected K": [
        best_k_cv,
        best_k_elbow,
        best_k_odd
    ],

    "Test Accuracy": [
        cv_test_accuracy,
        elbow_test_accuracy,
        odd_test_accuracy
    ]
})

print("\n========================================")
print("       KNN METHOD COMPARISON")
print("========================================")

print(comparison)

plt.figure(figsize=(10, 6))

plt.plot(
    results["K"],
    results["Accuracy"],
    marker="o",
    label="Test Accuracy"
)

plt.axvline(
    best_k_cv,
    linestyle="--",
    label=f"CV K = {best_k_cv}"
)

plt.axvline(
    best_k_elbow,
    linestyle="--",
    label=f"Elbow K = {best_k_elbow}"
)

plt.axvline(
    best_k_odd,
    linestyle="--",
    label=f"Odd K = {best_k_odd}"
)

plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy for Different K Values")
plt.legend()
plt.grid(True)

plt.show()
final_table = results.copy()

final_table["K Type"] = np.where(
    final_table["K"] % 2 == 1,
    "Odd",
    "Even"
)

print(final_table.to_string(index=False))