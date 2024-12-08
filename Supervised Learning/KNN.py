#sklearn: ML Library
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import matplotlib.pyplot as plt

# (1) Dataset Analysis
cancer =load_breast_cancer()
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target


# (2) Selection of The Machine Learning Model - KNN Classifier
# (3) Training The Model

X = cancer.data #features
y = cancer.target #target

# train test split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.3, random_state=42)

#Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Create And Train The KNN Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train) #The fit function trains the KNN algorithm using our data (samples + target).

# (4) Evaluation of Results: test
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

conf_matrix = confusion_matrix(y_test,y_pred)
print("Confusion matrix",conf_matrix)

# (5) Hyperparameter Tuning
'''
     KNN:Hyperparameter = K
     K:1,2,3 ... N
     Accuracy : %A , %B , %C ...
'''
accuracy_values = []
k_values = []
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)

plt.figure()
plt.plot(k_values,accuracy_values,marker ="o",linestyle="-")
plt.title("Accuracy based on the K value")    
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()