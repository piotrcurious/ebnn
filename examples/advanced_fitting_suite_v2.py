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
from ebnn.utils.aware_optimizer import AwareHamiltonianOptimizer
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
                loss += self.loss_func(self, yi, t[:, i, :])
            loss /= seq_len
        else:
            y = self.predictor(x)
            loss = self.loss_func(self, y, t)

        reporter.report({'loss': loss}, self)
        if x.ndim == 3:
            with chainer.no_backprop_mode():
                mse = 0
                for i in range(seq_len):
                    yi = self.predictor(x[:, i, :])
                    mse += F.mean_squared_error(yi, t[:, i, :])
                reporter.report({'mse': mse / seq_len}, self)
        else:
            reporter.report({'mse': F.mean_squared_error(y, t)}, self)

        return loss

def create_mlp(n_hidden=128):
    return chainer.Sequential(L.Linear(1, n_hidden), F.relu, L.Linear(n_hidden, n_hidden), F.relu, L.Linear(n_hidden, 1))

def create_lstm_predictor(out_size):
    return chainer.Sequential(
        BL.BinaryLSTM(1, out_size),
        L.Linear(None, 32), F.relu,
        L.Linear(32, 1)
    )

def train_and_evaluate(func_type, n_samples=1000, epochs=300):
    print(f"\n--- [Advanced Evaluation v2] {func_type} ---")
    train_data_raw, _ = get_dataset(func_type, n_samples=n_samples, noise=0.01)

    y_vals = np.array([p[1] for p in train_data_raw])
    y_min, y_max = y_vals.min(), y_vals.max()
    scale_func = lambda y: (y - y_min) / (y_max - y_min + 1e-7)
    unscale_func = lambda y: y * (y_max - y_min) + y_min

    train_data = [(x, scale_func(y).astype(np.float32).reshape(1)) for x, y in train_data_raw]

    seq_len = 10
    seq_train = []
    train_data_raw.sort(key=lambda x: x[0])
    for i in range(len(train_data_raw) - seq_len):
        x_seq = np.array([train_data_raw[j][0] for j in range(i, i+seq_len)]).astype(np.float32).reshape(seq_len, 1)
        y_seq = np.array([scale_func(train_data_raw[j][1]) for j in range(i, i+seq_len)]).astype(np.float32).reshape(seq_len, 1)
        seq_train.append((x_seq, y_seq))

    x_test = np.linspace(-1, 1, 100).astype(np.float32).reshape(-1, 1)
    y_test_true = np.array([p[1] for p in complex_function_generator(func_type, 100, noise=0, sorted_x=True)]).flatten()

    # Use Hamiltonian and Logical Loss
    # Ising encourages stable weight patterns, complexity encourages sparsity
    # We use lower weights for regularizers to prevent negative loss dominance
    loss_expr = "mse + 0.001 * complexity + 0.0001 * ising + 0.001 * state_transition"
    loss_func = BL.LossBuilder.build(loss_expr)

    methods = [
        ('MLP_Baseline', GenericModel(create_mlp(128), BL.MeanSquaredError()), False),
        ('LSTM_Aware_Hamiltonian', GenericModel(create_lstm_predictor(64), loss_func), True),
    ]

    results = {}
    for name, model, use_seq in methods:
        print(f"Training {name}...")
        batch_size = 32 if use_seq else 64
        it = chainer.iterators.SerialIterator(seq_train if use_seq else train_data, batch_size)

        lr = 0.001 if 'MLP' in name else 0.0005
        opt = chainer.optimizers.Adam(alpha=lr)
        if 'Aware' in name:
            opt = AwareHamiltonianOptimizer(opt)
        elif 'Logical' in name:
            opt = LogicalOptimizer(opt)
        opt.setup(model)

        out_dir = f'_tmp_v2_{func_type}_{name}'
        trainer = training.Trainer(training.StandardUpdater(it, opt), (epochs, 'epoch'), out=out_dir)
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/mse', 'elapsed_time']))
        trainer.run()

        with chainer.using_config('train', False):
            if use_seq:
                y_pred_scaled = []
                for xi in x_test:
                    if xi == x_test[0]:
                        for l in model.predictor:
                            if hasattr(l, 'reset_state'): l.reset_state()
                    y_p = model.predictor(xi.reshape(1, 1))
                    y_pred_scaled.append(y_p.data.flatten()[0])
                y_pred = unscale_func(np.array(y_pred_scaled))
            else:
                y_pred = unscale_func(model.predictor(x_test).data.flatten())
        results[name] = y_pred

    poly_coeffs = np.polyfit(np.array([p[0] for p in train_data_raw]).flatten(),
                             np.array([p[1] for p in train_data_raw]).flatten(), 12)
    results['Poly_Deg12'] = np.polyval(poly_coeffs, x_test.flatten())

    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_test_true, 'k--', label='True', alpha=0.5)
    for name, yp in results.items():
        mse_val = np.mean((yp - y_test_true)**2)
        plt.plot(x_test, yp, label=f'{name} (MSE: {mse_val:.4f})')
    plt.title(f'Evaluation v2: {func_type}')
    plt.legend(); plt.grid(True)
    path = f'final_eval_v2_{func_type}.png'
    plt.savefig(path); print(f"Saved {path}")
    plt.close()

if __name__ == '__main__':
    train_and_evaluate('fixed_poly', epochs=100)
    train_and_evaluate('fourier_series', epochs=150)
