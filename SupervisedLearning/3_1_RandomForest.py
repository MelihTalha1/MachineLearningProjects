from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt 

oli =fetch_olivetti_faces()

"""
   2D (64X64) -> 1D (4096)
"""
plt.figure()
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(oli.images[i], cmap = "gray")
    plt.axis("off")
plt.show()

X = oli.data
y = oli.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

rf_clf = RandomForestClassifier(n_estimators = 100, random_state= 42)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print("Acc: ",accuracy)




accuracy_values = []
rf_values = [5,50,100,500]
for rf in rf_values:
    raf = RandomForestClassifier(n_estimators=rf)
    raf.fit(X_train,y_train)
    y_pred = raf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    accuracy_values.append(accuracy)

plt.figure()
plt.plot(rf_values,accuracy_values,marker="o",linestyle ='-')
plt.title("Accuracy based on the RF value")
plt.xlabel("RF Value")
plt.ylabel("Accuracy")
plt.xticks(rf_values)
plt.grid(True)
plt.show()


