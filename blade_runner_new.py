#
import random

#
import numpy
import pandas
from matplotlib import pyplot
from scipy.stats import kendalltau
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import torch
from torch import nn


#
from neusk.neura import WrappedNN
from tests_new import check_up


#
data = pandas.read_csv('data/dataset.csv')
data = data.set_index(data.columns.values[0])


#
random.seed(999)
numpy.random.seed(999)
torch.manual_seed(999)
rs = 999


#


#
removables = [] + [] + ['Mjob', 'Fjob', 'reason', 'guardian']

target = 'G3'
x_factors = [x for x in data.columns if not any([y in x for y in [target] + removables])]

embeddings = ['sex', 'school', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid',
                'activities', 'nursery', 'higher', 'internet', 'romantic', ]
embeddings_ixs = [x_factors.index(e) for e in embeddings]
embedding_len_m = 4
embeddings_sizes = [(data[e].value_counts().shape[0], int(data[e].value_counts().shape[0] * embedding_len_m))
                    for e in embeddings]
# embeddings_sizes = [(data[e].value_counts().shape[0], int(data[e].value_counts().shape[0] ** (1 / 4)))
#                     for e in embeddings]

x_factors_numerical = [f for f in x_factors if f not in embeddings]
x_factors_categorical = embeddings

X_numerical = data[x_factors_numerical].values
X_categorical = data[x_factors_categorical].values
Y = data[target].values.astype(dtype=float)

ordinal = OrdinalEncoder()
ordinal_cols = embeddings
X_categorical = ordinal.fit_transform(X=X_categorical)

X_train_numerical, X_test_numerical, X_train_categorical, X_test_categorical, Y_train, Y_test = \
    train_test_split(X_numerical, X_categorical, Y, test_size=0.5, random_state=rs)

#
"""
thresh = 0.01
values = mutual_info_classif(X=X_train, y=Y_train, discrete_features='auto')
fs_mask = values >= thresh
"""
"""
thresh = 0.05
values = numpy.array([spearmanr(a=X_train[:, j], b=Y_train)[0] for j in range(X_train.shape[1])])
fs_mask = numpy.abs(values) >= thresh
"""
"""
thresh = 0.05
values = numpy.array([kendalltau(x=X_train[:, j], y=Y_train)[0] for j in range(X_train.shape[1])])
fs_mask = numpy.abs(values) >= thresh
"""
"""
alpha = 0.05
values = numpy.array([spearmanr(a=X_train[:, j], b=Y_train)[1] for j in range(X_train.shape[1])])
fs_mask = values <= alpha
"""
"""
alpha = 0.05
values = numpy.array([kendalltau(x=X_train[:, j], y=Y_train)[1] for j in range(X_train.shape[1])])
fs_mask = values <= alpha
"""
"""
X_train = X_train[:, fs_mask]
X_test = X_test[:, fs_mask]
"""


"""
scaler = StandardScaler()
scaler.fit(X=X_train)
X_train_ = scaler.transform(X_train)
X_test_ = scaler.transform(X_test)
"""
# """
X_train_numerical_ = X_train_numerical
X_train_categorical_ = X_train_categorical
X_test_numerical_ = X_test_numerical
X_test_categorical_ = X_test_categorical
# """

"""
# proj_rate = 0.50  # 0.75   0.5   0.25  'mle'
# njv = int(X_train_.shape[1] * proj_rate)
njv = 'mle'
# njv = 0.75  # 0.75  0.5  0.25
projector = PCA(n_components=njv, svd_solver='full', random_state=rs)
projector.fit(X=X_train_)
X_train_ = projector.transform(X_train_)
X_test_ = projector.transform(X_test_)
"""
"""
proj_rate = 0.50  # 0.75   0.5   0.25
gamma = 0.1  # None  0.001  0.1  1
njv = int(X_train_.shape[1] * proj_rate)
projector = KernelPCA(n_components=njv, random_state=rs, remove_zero_eig=True, gamma=gamma, kernel='rbf')
projector.fit(X=X_train_)
X_train_ = projector.transform(X_train_)
X_test_ = projector.transform(X_test_)
"""

Y_train_ = Y_train.reshape(-1, 1)
Y_test_ = Y_test.reshape(-1, 1)

X_train_numerical_ = torch.tensor(X_train_numerical_, dtype=torch.float)
X_train_categorical_ = torch.tensor(X_train_categorical_, dtype=torch.long)
Y_train_ = torch.tensor(Y_train_, dtype=torch.float)
X_test_numerical_ = torch.tensor(X_test_numerical_, dtype=torch.float)
X_test_categorical_ = torch.tensor(X_test_categorical_, dtype=torch.long)
Y_test_ = torch.tensor(Y_test_, dtype=torch.float)

nn_kwargs = {'layers': [nn.Linear, nn.Linear, nn.Linear, nn.Linear],
             'layers_dimensions': [256, 128, 64, 1],
             'layers_kwargs': [{}, {}, {}, {}],
             'embedding_sizes': embeddings_sizes,
             'embedding_indices': embeddings_ixs,
             'batchnorms': [None, nn.BatchNorm1d, nn.BatchNorm1d, None],  # nn.BatchNorm1d
             'activators': [None, nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU],
             'interdrops': [0.2, 0.2, 0.2, 0.0],
             'optimiser': torch.optim.Adamax,  # Adamax / AdamW / SGD
             'optimiser_kwargs': {'lr': 0.1,
                                  'weight_decay': 0.001,
                                  # 'momentum': 0.9,
                                  # 'nesterov': True,
                                  },
             # 'scheduler': torch.optim.lr_scheduler.ConstantLR,
             # 'scheduler_kwargs': {'factor': 1, 'total_iters': 1},
             # 'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
             # 'scheduler_kwargs': {'factor': 0.1, 'patience': 10, 'verbose': True},
             # 'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,
             # 'scheduler_kwargs': {'T_max': 100, 'eta_min': 0.001, 'verbose': True},
             'scheduler': torch.optim.lr_scheduler.CyclicLR,
             'scheduler_kwargs': {'base_lr': 0.001, 'max_lr': 0.1, 'step_size_up': 250, 'cycle_momentum': False,
                                  'mode': 'exp_range', 'gamma': 0.99, 'verbose': True},
             # 'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
             # 'scheduler_kwargs': {'T_0': 100, 'T_mult': 1, 'eta_min': 0.001, 'verbose': True},
             # 'weights_init': None,
             # 'weights_init_kwargs': None,
             # 'weights_init': nn.init.kaiming_uniform_,
             # 'weights_init_kwargs': {'mode': 'fan_in', 'nonlinearity': 'leaky_relu'},   # fan_in fan_out
             'weights_init': nn.init.kaiming_normal_,
             'weights_init_kwargs': {'mode': 'fan_in', 'nonlinearity': 'leaky_relu'},   # fan_in fan_out
             'loss_function': nn.MSELoss,
             'epochs': 3000
             #  'device': device,
             }


model = WrappedNN(**nn_kwargs)

model.fit(X_train_numerical=X_train_numerical_, X_train_categorical=X_train_categorical_, Y_train=Y_train_,
          X_val_numerical=X_test_numerical_, X_val_categorical=X_test_categorical_, Y_val=Y_test_)

y_hat_train = model.predict(X_numerical=X_train_numerical_, X_categorical=X_train_categorical_)
y_hat_test = model.predict(X_numerical=X_test_numerical_, X_categorical=X_test_categorical_)


class Mody:
    def __init__(self, model):
        self.model = model
    def fit(self, X_numerical, X_categorical, y):
        X_numerical = torch.tensor(X_numerical, dtype=torch.float)
        X_categorical = torch.tensor(X_categorical, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.float)
        self.model.fit(X_numerical_train=X_numerical, X_categorical_train=X_categorical, Y_train=y,
                       X_numerical_val=X_numerical, X_categorical_val=X_categorical, Y_val=y)
    def predict(self, X_numerical, X_categorical):
        X_numerical = torch.tensor(X_numerical, dtype=torch.float)
        X_categorical = torch.tensor(X_categorical, dtype=torch.long)
        return self.model.predict(X_numerical=X_numerical, X_categorical=X_categorical)


mody = Mody(model)
results_train = check_up(Y_train.flatten(), y_hat_train.flatten(), mody, X_train_numerical_)
results_test = check_up(Y_test.flatten(), y_hat_test.flatten(), mody, X_test_numerical_)

results_train['sample'] = 'train'
results_test['sample'] = 'test'

results_train = pandas.DataFrame(pandas.Series(results_train))
results_test = pandas.DataFrame(pandas.Series(results_test))
"""
# joblib.dump(model, filename='./model_ex12.pkl')
results_train.T.to_csv('./reported.csv', mode='a', header=False)
results_test.T.to_csv('./reported.csv', mode='a', header=False)

fig, ax = pyplot.subplots(2, 2, sharex='col', sharey='col')
ax[0, 0].plot(range(Y_train.shape[0]), Y_train.flatten() - y_hat_train.flatten(), color='navy')
ax[1, 0].plot(range(Y_test.shape[0]), Y_test.flatten() - y_hat_test.flatten(), color='orange')
ax[0, 1].hist(Y_train.flatten() - y_hat_train.flatten(), color='navy', bins=100, density=True)
ax[1, 1].hist(Y_test.flatten() - y_hat_test.flatten(), color='orange', bins=100, density=True)

"""