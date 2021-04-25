# Sample Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
##from sklearn.tree import DecisionTreeClassifier
##from sklearn.naive_bayes import GaussianNB
##import statsmodels.api as sm
##from sklearn.linear_model import LinearRegression
##from sklearn.linear_model import LogisticRegression
##from sklearn import svm
##from sklearn.ensemble import BaggingClassifier
##from sklearn.ensemble import AdaBoostClassifier #AdaBoost es la 
#abreviatura de adaptive boosting, es un algoritmo que puede ser utilizado junto 
#con otros algoritmos de aprendizaje para mejorar su rendimiento.  AdaBoost funciona 
#eligiendo un algoritmo base (por ejemplo árboles de decisión) y mejorándolo iterativamente 
#al tomar en cuenta los casos incorrectamente clasificados en el conjunto de entrenamiento.
##from sklearn.cluster import KMeans
##from sklearn.cluster import AgglomerativeClustering
##from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture# Un Gaussian Mixture model es un modelo probabilístico
# en el que se considera que las observaciones siguen una distribución probabilística formada por 
#la combinación de múltiples distribuciones normales (componentes). En su aplicación al clustering,
# puede entenderse como una generalización de K-means con la que, en lugar de asignar cada observación 
#a un único cluster, se obtiene una probabilidad de pertenencia a cada uno.
##from sklearn.decomposition import PCA
##from sklearn.preprocessing import StandardScaler#para PCA

# load the iris datasets
dataset = datasets.load_iris()
# fit a CART model to the data

#model = DecisionTreeClassifier()#con esto elegimos un algoritmo de ML
#model =GaussianNB()#Naive Bayes Algorithms
#model= LogisticRegression() #LOGISTIC REGRESSION
#model= svm.SVC()#Support Vector Machines 
#model= BaggingClassifier()# metodos ensemble
# Utilizando AdaBoost para aumentar la precisión
model = AdaBoostClassifier()


model.fit(dataset.data, dataset.target)# Con esto lo entrenamos en los datos.
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
    