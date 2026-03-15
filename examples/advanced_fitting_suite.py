import os
import sys
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, reporter
from chainer.training import extensions
import matplotlib.pyplot as plt

# Setup path to import ebnn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ebnn.links as BL
from dataset_generator import get_dataset

class GenericModel(chainer.Chain):
    def __init__(self, predictor, loss_func):
        super(GenericModel, self).__init__()
        with self.init_scope():
            self.predictor = predictor
        self.loss_func = loss_func

    def __call__(self, x, t):
        if hasattr(self.predictor[0], 'reset_state'):
            self.predictor[0].reset_state()
        y = self.predictor(x)
        loss = self.loss_func(y, t)
        reporter.report({'loss': loss}, self)
        return loss

def create_mlp(n_hidden=128):
    return chainer.Sequential(
        L.Linear(1, n_hidden),
        F.relu,
        L.Linear(n_hidden, n_hidden),
        F.relu,
        L.Linear(n_hidden, 1)
    )

def create_lstm_predictor(out_size, max_complexity):
    return chainer.Sequential(
        BL.BinaryLSTM(1, out_size, max_complexity=max_complexity),
        L.Linear(None, 1)
    )

def train_and_evaluate(func_type, n_samples=2000):
    print(f"\n--- Evaluating Function Type: {func_type} ---")
    train_data, test_data = get_dataset(func_type, n_samples=n_samples, noise=0.05)

    x_test = np.array([d[0] for d in test_data]).flatten()
    y_test = np.array([d[1] for d in test_data]).flatten()
    x_train = np.array([d[0] for d in train_data]).flatten()
    y_train = np.array([d[1] for d in train_data]).flatten()

    methods = [
        ('NN_MSE', GenericModel(create_mlp(128), BL.MeanSquaredError())),
        ('NN_Huber', GenericModel(create_mlp(128), BL.HuberLoss())),
        ('LSTM_C16', GenericModel(create_lstm_predictor(32, 16), BL.MeanSquaredError())),
        ('LSTM_C8', GenericModel(create_lstm_predictor(32, 8), BL.MeanSquaredError())),
    ]

    results = {}

    for name, model in methods:
        print(f"Training {name}...")
        train_iter = chainer.iterators.SerialIterator(train_data, 64)
        optimizer = chainer.optimizers.Adam(alpha=0.001)
        optimizer.setup(model)
        updater = training.StandardUpdater(train_iter, optimizer)
        trainer = training.Trainer(updater, (100, 'epoch'), out=f'_tmp_{func_type}_{name}')
        trainer.run()

        x_test_var = chainer.Variable(x_test.reshape(-1, 1).astype(np.float32))
        if hasattr(model.predictor[0], 'reset_state'):
            model.predictor[0].reset_state()
        y_pred = model.predictor(x_test_var).data.flatten()
        results[name] = y_pred

    # Classical Polynomial Fit (Degree 7)
    poly_coeffs = np.polyfit(x_train, y_train, 7)
    y_poly_pred = np.polyval(poly_coeffs, x_test)
    results['Classical_Poly7'] = y_poly_pred

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.scatter(x_test, y_test, label='True Data (noisy)', alpha=0.15, color='gray')
    idx = np.argsort(x_test)

    for name, y_pred in results.items():
        mse = np.mean((y_pred - y_test)**2)
        plt.plot(x_test[idx], y_pred[idx], label=f'{name} (MSE: {mse:.4f})', linewidth=2)

    plt.title(f'Fitting comparison for {func_type}')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plot_path = f'eval_{func_type}.png'
    plt.savefig(plot_path)
    print(f"Graph saved as {plot_path}")

    # Cleanup
    import shutil
    for name, _ in methods:
        shutil.rmtree(f'_tmp_{func_type}_{name}', ignore_errors=True)

if __name__ == '__main__':
    for ft in ['compound_sine', 'sine_exp_decay', 'varying_noise', 'polynomial']:
        train_and_evaluate(ft)
