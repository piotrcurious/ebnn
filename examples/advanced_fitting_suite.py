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
from ebnn.utils.logical_optimizer import LogicalOptimizer
from dataset_generator import get_dataset, complex_function_generator

class GenericModel(chainer.Chain):
    def __init__(self, predictor, loss_func):
        super(GenericModel, self).__init__()
        with self.init_scope():
            self.predictor = predictor
        self.loss_func = loss_func

    def __call__(self, x, t):
        for link in self.predictor:
            if hasattr(link, 'reset_state'): link.reset_state()

        if x.ndim == 3: # Sequence
            loss = 0; seq_len = x.shape[1]
            for i in range(seq_len):
                yi = self.predictor(x[:, i, :])
                # We need to pass the model (self) to support state/complexity losses
                loss += self.loss_func(self, yi, t[:, i, :])
            loss /= seq_len
        else:
            y = self.predictor(x)
            loss = self.loss_func(self, y, t)

        reporter.report({'loss': loss}, self)
        return loss

def create_mlp(n_hidden=256):
    return chainer.Sequential(L.Linear(1, n_hidden), F.relu, L.Linear(n_hidden, n_hidden), F.relu, L.Linear(n_hidden, 1))

def create_lstm_predictor(out_size):
    return chainer.Sequential(
        BL.BinaryLSTM(1, out_size),
        L.Linear(None, 64), F.relu,
        L.Linear(64, 1)
    )

def train_and_evaluate(func_type, n_samples=2000, epochs=100):
    print(f"\n--- [Final Evaluation] {func_type} ---")
    train_data, _ = get_dataset(func_type, n_samples=n_samples, noise=0.01)
    seq_train, _ = get_dataset(func_type, n_samples=n_samples, noise=0.01, sequence=True, seq_len=10)

    y_raw = np.array([p[1] for p in train_data])
    y_mean, y_std = y_raw.mean(), y_raw.std()
    norm = lambda y: (y - y_mean) / (y_std + 1e-7)
    train_data = [(x, norm(y)) for x, y in train_data]
    seq_train = [(x, norm(y)) for x, y in seq_train]

    x_test = np.linspace(-1, 1, 100).astype(np.float32).reshape(-1, 1)
    y_test_true = norm(np.array([p[1] for p in complex_function_generator(func_type, 100, noise=0, sorted_x=True)]).flatten())

    # Build loss using LossBuilder
    loss_expr = "mse + 0.1 * equiv + 0.01 * state_transition"
    loss_func = BL.LossBuilder.build(loss_expr)

    methods = [
        ('MLP_Baseline', GenericModel(create_mlp(256), BL.MeanSquaredError()), False),
        ('LSTM_Logical_Final', GenericModel(create_lstm_predictor(64), loss_func), True),
    ]

    results = {}
    for name, model, use_seq in methods:
        print(f"Training {name}...")
        it = chainer.iterators.SerialIterator(seq_train if use_seq else train_data, 64)
        opt = chainer.optimizers.Adam(alpha=0.001)
        if 'Logical' in name: opt = LogicalOptimizer(opt)
        opt.setup(model)

        trainer = training.Trainer(training.StandardUpdater(it, opt), (epochs, 'epoch'), out=f'_tmp_final_{func_type}_{name}')
        trainer.run()

        with chainer.using_config('train', False):
            if use_seq:
                for l in model.predictor:
                    if hasattr(l, 'reset_state'): l.reset_state()
                y_pred = np.array([model.predictor(xi.reshape(1,1)).data.flatten()[0] for xi in x_test])
            else:
                y_pred = model.predictor(x_test).data.flatten()
        results[name] = y_pred

    poly_coeffs = np.polyfit(np.array([p[0] for p in train_data]).flatten(),
                             np.array([p[1] for p in train_data]).flatten(), 12)
    results['Poly_Deg12'] = np.polyval(poly_coeffs, x_test.flatten())

    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_test_true, 'k--', label='True', alpha=0.5)
    for name, yp in results.items():
        mse_val = np.mean((yp - y_test_true)**2)
        plt.plot(x_test, yp, label=f'{name} (MSE: {mse_val:.4f})')
    plt.title(f'Final Result: {func_type}')
    plt.legend(); plt.grid(True)
    path = f'final_eval_{func_type}.png'
    plt.savefig(path); print(f"Saved {path}")

if __name__ == '__main__':
    train_and_evaluate('fourier_series', epochs=100)
    train_and_evaluate('fixed_poly', epochs=100)
