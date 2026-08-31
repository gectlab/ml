import pandas as pd
import numpy as np

data = pd.read_csv("id3.csv")
target = "Play"
features = ["Outlook", "Temperature", "Humidity", "Windy"]


def entropy(y):

    classes, counts = np.unique(y, return_counts=True)

    probabilities = counts / len(y)

    entropy_value = 0

    for p in probabilities:
        if p > 0:
            entropy_value -= p * np.log2(p)

    return entropy_value


def information_gain(data, feature, target):

    parent_entropy = entropy(data[target].values)

    values, counts = np.unique(
        data[feature],
        return_counts=True
    )

    weighted_entropy = 0

    for value, count in zip(values, counts):

        subset = data[data[feature] == value]

        subset_entropy = entropy(
            subset[target].values
        )

        weight = count / len(data)

        weighted_entropy += weight * subset_entropy

    gain = parent_entropy - weighted_entropy

    return gain

def best_feature(data, features, target):

    gains = {}

    for feature in features:

        gains[feature] = information_gain(
            data,
            feature,
            target
        )

    best = max(gains, key=gains.get)

    return best



def build_tree(data, features, target):

    if len(np.unique(data[target])) == 1:

        return data[target].iloc[0]

    if len(features) == 0:

        return data[target].mode()[0]

    best = best_feature(
        data,
        features,
        target
    )

    tree = {best: {}}

    remaining_features = [
        feature
        for feature in features
        if feature != best
    ]

    for value in np.unique(data[best]):

        subset = data[data[best] == value]

        tree[best][value] = build_tree(
            subset,
            remaining_features,
            target
        )

    return tree

tree = build_tree(
    data,
    features,
    target
)

print("\nDecision Tree:")
print(tree)