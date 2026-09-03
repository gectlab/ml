import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

# Load data
data = pd.read_csv("naive.csv")
X_raw = data["Message"]
Y = data["Class"]

# Split dataset
X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(
    X_raw, Y, test_size=0.2, random_state=42
)

# 1. Multinomial Naive Bayes
# Extract word counts
count_vec = CountVectorizer()
Xm_train = count_vec.fit_transform(X_train_raw)
Xm_test = count_vec.transform(X_test_raw)

# Train & Predict
mnb = MultinomialNB()
mnb.fit(Xm_train, Y_train)
Y_pred_multi = mnb.predict(Xm_test)

print("Multinomial Predictions:")
print("Actual:   ", Y_test.values)
print("Predicted:", Y_pred_multi)
print(
    classification_report(Y_test, Y_pred_multi, pos_label="Spam", zero_division=0)
)

# 2. Bernoulli Naive Bayes
# Extract binary presence indicators
bern_vec = CountVectorizer(binary=True)
Xb_train = bern_vec.fit_transform(X_train_raw)
Xb_test = bern_vec.transform(X_test_raw)

# Train & Predict
bnb = BernoulliNB()
bnb.fit(Xb_train, Y_train)
Y_pred_bernoulli = bnb.predict(Xb_test)

print("\nBernoulli Predictions:")
print("Actual:   ", Y_test.values)
print("Predicted:", Y_pred_bernoulli)
print(
    classification_report(
        Y_test, Y_pred_bernoulli, pos_label="Spam", zero_division=0
    )
)
