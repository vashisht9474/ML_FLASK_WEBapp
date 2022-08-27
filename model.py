import pandas as pd
import joblib


from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("iris.csv")

X = df[["SepalLengthCm", "SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y = df["Species"]

clf = GaussianNB() 
clf.fit(X, y)
joblib.dump(clf, "clf.pkl")
