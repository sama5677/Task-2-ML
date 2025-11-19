import math

# =========================
# KNN Implementation
# =========================

data_knn = [
    (2, 8, "Failed"),
    (3, 7, "Failed"),
    (4, 6, "Failed"),
    (5, 7, "Passed"),
    (6, 8, "Passed"),
    (7, 6, "Passed"),
    (8, 5, "Failed"),
    (9, 7, "Passed"),
    (10, 8, "Passed"),
    (11, 6, "Passed")
]

# Euclidean distance
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def knn_predict(new_point, k=3):
    distances = []
    for study, sleep, result in data_knn:
        d = distance(new_point, (study, sleep))
        distances.append((d, result))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    passed = sum(1 for _, r in k_nearest if r == "Passed")
    failed = k - passed

    return "Passed" if passed > failed else "Failed"


# Example new student
new_student = (6, 7)
print("KNN Prediction for student (6 study, 7 sleep):", knn_predict(new_student))


# =========================
# Naive Bayes Implementation
# =========================

data_nb = [
    ("Sunny", "Hot", "High", False, "No"),
    ("Sunny", "Hot", "High", True, "No"),
    ("Overcast", "Hot", "High", False, "Yes"),
    ("Rainy", "Mild", "High", False, "Yes"),
    ("Rainy", "Cool", "Normal", False, "Yes"),
    ("Rainy", "Cool", "Normal", True, "No"),
    ("Overcast", "Cool", "Normal", True, "Yes"),
    ("Sunny", "Mild", "High", False, "No"),
    ("Sunny", "Cool", "Normal", False, "Yes"),
    ("Rainy", "Mild", "Normal", False, "Yes"),
    ("Sunny", "Mild", "Normal", True, "Yes"),
    ("Overcast", "Mild", "High", True, "Yes"),
    ("Overcast", "Hot", "Normal", False, "Yes"),
    ("Rainy", "Mild", "High", True, "No")
]

# Function to compute probability
def naive_bayes_predict(outlook, temp, humidity, windy):
    yes_count = sum(1 for row in data_nb if row[4] == "Yes")
    no_count = sum(1 for row in data_nb if row[4] == "No")
    total = len(data_nb)

    P_yes = yes_count / total
    P_no = no_count / total

    def cond_prob(feature_index, value, label):
        label_rows = [row for row in data_nb if row[4] == label]
        count = sum(1 for row in label_rows if row[feature_index] == value)
        return count / len(label_rows) if len(label_rows) > 0 else 0

    # Likelihoods
    yes_likelihood = (
        cond_prob(0, outlook, "Yes") *
        cond_prob(1, temp, "Yes") *
        cond_prob(2, humidity, "Yes") *
        cond_prob(3, windy, "Yes") * P_yes
    )

    no_likelihood = (
        cond_prob(0, outlook, "No") *
        cond_prob(1, temp, "No") *
        cond_prob(2, humidity, "No") *
        cond_prob(3, windy, "No") * P_no
    )

    return "Yes" if yes_likelihood > no_likelihood else "No"


# Example new day case
new_day = ("Sunny", "Mild", "High", False)
result = naive_bayes_predict(*new_day)
print("Naive Bayes prediction for new day:", result)