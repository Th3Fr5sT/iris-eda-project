import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


iris = load_iris(as_frame=True)
df = iris.frame
#print(df)

# Replace numeric target with species names
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

print("First rows of dataset:\n", df.head(), "\n")
print("Class names:", iris.target_names, "\n")
print("------------------------------------------------")

# Petal Length vs Petal Width
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df, x="petal length (cm)", y="petal width (cm)",
    hue="species", palette="Set1", s=70, alpha=0.7
)
plt.title("Petal Length vs Petal Width")
plt.show()

# Sepal Length vs Sepal Width
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df, x="sepal length (cm)", y="sepal width (cm)",
    hue="species", palette="Set2", s=70, alpha=0.7
)
plt.title("Sepal Length vs Sepal Width")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.drop(["target","species"], axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Iris Features")
plt.show()

# Train-test split
X = df.drop(["target","species"], axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape, "\n")

# Standardize numeric features
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

scaled_range = pd.DataFrame({
    "min": X_train_scaled.min(),
    "max": X_train_scaled.max()
})

print("Range of scaled features:\n", scaled_range)

print("------------------------------------------------")
print("Scaled training features (first rows):\n", X_train_scaled.head(), "\n")

