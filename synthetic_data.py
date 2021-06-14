from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def generate_random_dataset(n=10000, n_useless=5):
    age_ = np.linspace(45, 90, 50)
    bmi_ = np.arange(17, 34)

    def get_lky(age, bmi):
        return (age - 65) / 10 + abs((bmi - 24) / 5.5)

    X_useless = [np.random.normal(size=n) for _ in range(n_useless)]
    X = np.column_stack([
         np.random.uniform(age_[0], age_[-1], size=n),
         np.random.uniform(bmi_[0], bmi_[-1], size=n)
    ] + X_useless)
    y = (get_lky(X[:, 0], X[:, 1]) > 1.1).astype('int')
    return X, y

from statistics import mode


def plot_decision_boundary(est, X, y):
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    h = 100

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    x_axis, y_axis = (
        np.linspace(x_min, x_max, h),
        np.linspace(y_min, y_max, h)
    )
    xx, yy = np.meshgrid(x_axis, y_axis)
    xxx, yyy = xx.ravel(), yy.ravel()
    prob = est.predict_proba(
        np.column_stack(
            [xxx, yyy] + [np.zeros_like(xxx)] * (X.shape[1] - 2)
        )
    )[:, 1]
    prob = prob.reshape(len(x_axis), len(y_axis))
    plt.contourf(x_axis, y_axis, prob, cmap=cm, alpha=.8)
    plt.scatter(X[:, 0], X[:, 1],  c=y, cmap=cm_bright, edgecolors='k')


def generate_test_point(age, bmi, n_useless):
    return np.concatenate(
        [np.array([age]), np.array([bmi])] + [np.random.normal(size=1) for _ in range(n_useless)]
    ).reshape(1, -1)


def plot_test_point(X, c='k'):
    size = 120
    plt.scatter(X[:, 0], X[:, 1], c=c, marker='x', s=size, linewidth=5)

################################################################################
# first columns represents age and second bmi
RANDOM_SEED = 5000
np.random.seed(RANDOM_SEED)
N_USELESS = 5

# generate a dataset with two useful features (age and bmi) and 5 useless features
X, y = generate_random_dataset(n_useless=N_USELESS)

# No te preocupes por el clasificador, simplemente úsalo como hasta ahora
est = RandomForestClassifier()
est.fit(X, y)



plot_decision_boundary(est, X[:1000], y[:1000])
plt.xlabel('Age')
plt.ylabel('BMI')

# Generate a point with age 74 and bmi 24
X_sim = generate_test_point(85,32, n_useless=N_USELESS)
print(X_sim)
# plot it to check where does it fall...
plot_test_point(X_sim, c='k')

#import lime.lime_tabular
#explainer = lime.lime_tabular.LimeTabularExplainer(X)
#exp = explainer.explain_instance(X_sim.flatten(), est.predict_proba)
#fig = exp.as_pyplot_figure()
#fig.show()
import shap
X=np.array(X)
explainer = shap.KernelExplainer(est.predict, shap.kmeans(X, 100))
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[X_sim,:], X[X_sim,:])
# We can repeat for several data points
data = [
    (76, 24, 'k'), (65, 29, 'g'), (65, 30.5, 'g'), (70, 20.5, 'g'),
    (70, 21.5, 'g'), (50, 25, 'w'), (85, 32, 'w')
]
#for age, bmi, color in data:
#    X_sim = generate_test_point(age, bmi, n_useless=N_USELESS)
#    plot_test_point(X_sim, c=color)
#plt.show()

