import numpy as np
import pandas as pd

data=pd.read_csv("naive.csv")

messages=data["Message"].values
Y=data["Class"].values

vocabulary=[]
for message in messages:
    words=message.lower().split()
    for word in words:
        if word not in vocabulary:
            vocabulary.append(word)

X_bernoulli=[]
for message in messages:
    words=message.lower().split()
    row=[]
    for word in vocabulary:
        if word in words:
            row.append(1)
        else:
            row.append(0)
    X_bernoulli.append(row)
X_bernoulli=np.array(X_bernoulli)

X_multinomial=[]
for message in messages:
    words=message.lower().split()
    row=[]
    for word in vocabulary:
        row.append(words.count(word))
    X_multinomial.append(row)
X_multinomial=np.array(X_multinomial)

np.random.seed(42)
indices=np.random.permutation(len(X_bernoulli))
train_size=int(0.8*len(indices))
train_idx=indices[:train_size]
test_idx=indices[train_size:]

Xb_train=X_bernoulli[train_idx]
Xb_test=X_bernoulli[test_idx]
Xm_train=X_multinomial[train_idx]
Xm_test=X_multinomial[test_idx]
Y_train=Y[train_idx]
Y_test=Y[test_idx]

classes=np.unique(Y_train)
priors={}
for c in classes:
    priors[c]=np.sum(Y_train==c)/len(Y_train)

word_probabilities={}
V=len(vocabulary)
for c in classes:
    class_data=Xm_train[Y_train==c]
    words_counts=np.sum(class_data,axis=0)
    total_words=np.sum(words_counts)
    word_probabilities[c]=(words_counts+1)/(total_words+V)

def multinomial_predict(X_test,classes,priors,word_probabilities):
    predictions=[]
    for row in X_test:
        scores={}
        for c in classes:
            score=priors[c]
            for i in range(len(row)):
                score*=word_probabilities[c][i]**row[i]
            scores[c]=score
        prediction=max(scores,key=scores.get)
        predictions.append(prediction)
    return np.array(predictions)
Y_pred_multi=multinomial_predict(Xm_test,classes,priors,word_probabilities)
print("Multinomial Predictions:")
print("Actual:",Y_test)
print("Predicted:",Y_pred_multi)

TP=np.sum((Y_pred_multi=="Spam")&(Y_test=="Spam"))
TN=np.sum((Y_pred_multi=="Ham")&(Y_test=="Ham"))
FP=np.sum((Y_pred_multi=="Spam")&(Y_test=="Ham"))
FN=np.sum((Y_pred_multi=="Ham")&(Y_test=="Spam"))

acc=(TP+TN)/(TP+TN+FP+FN)
pre=TP/(TP+FP)
recall=TP/(TP+FN)
f1=(2*pre*recall)/(pre+recall)

print("Accuracy:",acc)
print("Precision:",pre)
print("Recall:",recall)
print("F1:",f1)

bernoulli_probabilities={}
for c in classes:
    class_data=Xb_train[Y_train==c]
    word_present=np.sum(class_data,axis=0)
    n=len(class_data)
    p_present=(word_present+1)/(n+2)
    p_absent=1-p_present
    bernoulli_probabilities[c]=(p_present,p_absent)

def bernoulli_predict(X_test,classes,priors,probabilities):
    predictions=[]
    for row in X_test:
        scores={}
        for c in classes:
            p_present,p_absent=probabilities[c]
            score=priors[c]
            for i in range(len(row)):
                if row[i]==1:
                    score*=p_present[i]
                else:
                    score*=p_absent[i]
            scores[c]=score
        prediction=max(scores,key=scores.get)
        predictions.append(prediction)
    return np.array(predictions)
Y_pred_bernoulli=bernoulli_predict(Xb_test,classes,priors,bernoulli_probabilities)
print("Bernoulli Predictions:")
print("Actual",Y_test)
print("Predicted",Y_pred_bernoulli)

TP=np.sum((Y_pred_bernoulli=="Spam")&(Y_test=="Spam"))
TN=np.sum((Y_pred_bernoulli=="Ham")&(Y_test=="Ham"))
FP=np.sum((Y_pred_bernoulli=="Spam")&(Y_test=="Ham"))
FN=np.sum((Y_pred_bernoulli=="Ham")&(Y_test=="Spam"))

acc=(TP+TN)/(TP+TN+FP+FN)
pre=TP/(TP+FP)
recall=TP/(TP+FN)
f1=(2*pre*recall)/(pre+recall)

print("Accuracy:",acc)
print("Precision:",pre)
print("Recall:",recall)
print("F1:",f1)