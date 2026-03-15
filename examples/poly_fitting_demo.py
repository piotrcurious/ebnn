import numpy as np
import chainer
from chainer import datasets
import matplotlib.pyplot as plt
import os
import sys

# Add the root directory to sys.path to import ebnn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ebnn.links as BL
import chainer.functions as F
import chainer.links as L
from chainer import reporter, training
from chainer.training import extensions

def poly_generator(degree=3, n_samples=1000, noise=0.1):
    coeffs = np.random.randn(degree + 1)
    x = np.random.uniform(-1, 1, n_samples).astype(np.float32)
    y = np.polyval(coeffs[::-1], x).astype(np.float32)
    y += np.random.normal(0, noise, n_samples).astype(np.float32)
    for i in range(n_samples):
        yield x[i:i+1], y[i:i+1]

class PolyFittingModel(chainer.Chain):
    def __init__(self, n_hidden=32, loss_type='mse'):
        super(PolyFittingModel, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(1, n_hidden)
            self.l2 = L.Linear(n_hidden, n_hidden)
            self.l3 = L.Linear(n_hidden, 1)

            if loss_type == 'mse':
                self.loss_func = BL.MeanSquaredError()
            elif loss_type == 'mae':
                self.loss_func = BL.AbsoluteError()
            else:
                self.loss_func = BL.MeanSquaredError()

    def __call__(self, x, t):
        y = self.predict(x)
        loss = self.loss_func(y, t)
        reporter.report({'loss': loss}, self)
        return loss

    def predict(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)

class LSTMPolyModel(chainer.Chain):
    def __init__(self, out_size=16, max_complexity=None):
        super(LSTMPolyModel, self).__init__()
        with self.init_scope():
            self.lstm = BL.BinaryLSTM(1, out_size, max_complexity=max_complexity)
            self.l1 = L.Linear(None, 1)
        self.loss_func = BL.MeanSquaredError()

    def __call__(self, x, t):
        self.lstm.reset_state()
        y = self.predict(x)
        loss = self.loss_func(y, t)
        reporter.report({'loss': loss}, self)
        return loss

    def predict(self, x):
        h = self.lstm(x)
        return self.l1(h)

def run_experiment():
    degree = 3
    n_samples = 1000
    data_gen = poly_generator(degree, n_samples)
    dataset = list(data_gen)
    train_data = dataset[:800]
    test_data = dataset[800:]

    x_train = np.array([d[0] for d in train_data]).flatten()
    y_train = np.array([d[1] for d in train_data]).flatten()
    x_test = np.array([d[0] for d in test_data]).flatten()
    y_test = np.array([d[1] for d in test_data]).flatten()

    results = {}

    for name, model in [('NN_MSE', PolyFittingModel(loss_type='mse')),
                        ('LSTM_C8', LSTMPolyModel(out_size=16, max_complexity=8))]:
        print(f"Training {name}...")
        train_iter = chainer.iterators.SerialIterator(train_data, 32)
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)
        updater = training.StandardUpdater(train_iter, optimizer)
        trainer = training.Trainer(updater, (30, 'epoch'), out=f'_tmp_{name}')
        trainer.run()

        x_test_var = chainer.Variable(x_test.reshape(-1, 1).astype(np.float32))
        if hasattr(model, 'lstm'): model.lstm.reset_state()
        y_model_pred = model.predict(x_test_var).data.flatten()
        results[name] = (y_model_pred, np.mean((y_model_pred - y_test)**2))

    classical_coeffs = np.polyfit(x_train, y_train, degree)
    y_classical_pred = np.polyval(classical_coeffs, x_test)
    classical_mse = np.mean((y_classical_pred - y_test)**2)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_test, y_test, label='True Data', alpha=0.3)
    idx = np.argsort(x_test)
    plt.plot(x_test[idx], y_classical_pred[idx], label=f'Classical Fit (MSE: {classical_mse:.4f})')
    for name, (pred, mse) in results.items():
        plt.plot(x_test[idx], pred[idx], label=f'{name} (MSE: {mse:.4f})')
    plt.legend()
    plt.title('Polynomial Fitting Comparison')
    plt.savefig('poly_fitting_eval.png')
    import shutil
    for name in ['NN_MSE', 'LSTM_C8']: shutil.rmtree(f'_tmp_{name}', ignore_errors=True)

if __name__ == '__main__':
    run_experiment()
