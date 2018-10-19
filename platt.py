# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import numpy as np

np.random.seed(123)

import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.optimize import fmin_bfgs
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
from sklearn.base import TransformerMixin


class PlattScaler(TransformerMixin):
    """ Perform Platt Scaling.
    Based on Platt 1999
    """

    def __init__(self):
        pass

    def fit(self, f, y):
        """ Fit Platt model.
        This method takes in the classifier outputs and the true labels,
        and fits a scaling model to convert classifier outputs to true
        probabilities. Sticks with Platt's weird notation throughout.

            f: classifier outputs
            y: true labels
        """
        eps = np.finfo(np.float).tiny  # to avoid division by 0 warning

        # Bayes priors
        prior0 = float(np.sum(y <= 0))
        prior1 = y.shape[0] - prior0
        T = np.zeros(y.shape)
        T[y > 0] = (prior1 + 1.) / (prior1 + 2.)
        T[y <= 0] = 1. / (prior0 + 2.)
        T1 = 1. - T

        def objective(theta):
            A, B = theta
            E = np.exp(A * f + B)
            P = 1. / (1. + E)
            l = -(T * np.log(P + eps) + T1 * np.log(1. - P + eps))
            return l.sum()

        def grad(theta):
            A, B = theta
            E = np.exp(A * f + B)
            P = 1. / (1. + E)
            TEP_minus_T1P = P * (T * E - T1)
            dA = np.dot(TEP_minus_T1P, f)
            dB = np.sum(TEP_minus_T1P)
            return np.array([dA, dB])

        AB0 = np.array([0., np.log((prior0 + 1.) / (prior1 + 1.))])
        self.A_, self.B_ = fmin_bfgs(objective, AB0, fprime=grad, disp=False)


    def transform(self, f):
        """
        Given a set of classifer outputs return probs.
        """
        return 1. / (1. + np.exp(self.A_ * f + self.B_))

    def fit_transform(self, f, y):
        self.fit(f, y)
        return self.transform(f)




def plot_calibration_curve(y, p):

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y, p, n_bins=10
    )

    plt.figure()
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y, p, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-")

    ax2.hist(p, range=(0, 1), bins=10,
             histtype="stepfilled", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([0, 1.])
    ax1.set_xlim([0, 1.])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    X, y = datasets.make_classification(
        n_samples=100000, n_features=20, n_informative=2, n_redundant=2
    )

    train_samples = 100  # Samples used for training the models

    X_train = X[:train_samples]
    X_test = X[train_samples:]
    y_train = y[:train_samples]
    y_test = y[train_samples:]


    svc = LinearSVC(C=1.0)
    pls = PlattScaler()


    svc.fit(X_train, y_train)
    df_train = svc.decision_function(X_train)

    pls.fit(df_train, y_train)

    df_test = svc.decision_function(X_test)

    prob_pos = pls.transform(df_test)
    plot_calibration_curve(y_test, prob_pos)
