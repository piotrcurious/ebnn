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
from ebnn.utils.logical_optimizer import LogicalOptimizer, LogicalWeightScheduler
from dataset_generator import get_dataset, complex_function_generator

class GenericModel(chainer.Chain):
    def __init__(self, predictor, loss_func, state_loss=None, optimizer=None):
        super(GenericModel, self).__init__()
        with self.init_scope():
            self.predictor = predictor
        self.loss_func = loss_func
        self.state_loss = state_loss
        self.optimizer = optimizer

    def __call__(self, x, t):
        for link in self.predictor:
            if hasattr(link, 'reset_state'):
                link.reset_state()

        logical_weight = 0.1
        if self.optimizer is not None and hasattr(self.optimizer, 'logical_weight'):
            logical_weight = self.optimizer.logical_weight

        if x.ndim == 3: # Sequence training
            loss = 0
            seq_len = x.shape[1]
            for i in range(seq_len):
                yi = self.predictor(x[:, i, :])
                ti = t[:, i, :]

                if isinstance(self.loss_func, BL.CompositeLoss):
                    current_loss = 0
                    for lf, w in self.loss_func.losses_with_weights:
                        if 'Logical' in lf.__class__.__name__:
                            current_loss += w * logical_weight * lf(yi, ti)
                        else:
                            current_loss += w * lf(yi, ti)
                    loss += current_loss
                else:
                    loss += self.loss_func(yi, ti)

                if self.state_loss:
                    loss += logical_weight * self.state_loss(self, x[:, i, :], ti)
            loss /= seq_len
        else:
            y = self.predictor(x)
            if isinstance(self.loss_func, BL.CompositeLoss):
                loss = 0
                for lf, w in self.loss_func.losses_with_weights:
                    if 'Logical' in lf.__class__.__name__:
                        loss += w * logical_weight * lf(y, t)
                    else:
                        loss += w * lf(y, t)
            else:
                loss = self.loss_func(y, t)

            if self.state_loss:
                loss += logical_weight * self.state_loss(self, x, t)

        reporter.report({'loss': loss}, self)
        return loss

def create_mlp(n_hidden=128):
    return chainer.Sequential(
        L.Linear(1, n_hidden), F.relu,
        L.Linear(n_hidden, n_hidden), F.relu,
        L.Linear(n_hidden, 1)
    )

def create_lstm_predictor(out_size, max_complexity):
    return chainer.Sequential(
        BL.BinaryLSTM(1, out_size, max_complexity=max_complexity),
        L.Linear(None, 32), F.relu,
        L.Linear(32, 1)
    )

def train_and_evaluate(func_type, n_samples=2000, epochs=100):
    print(f"\n--- [V4 Iteration] Evaluating: {func_type} ---")

    train_data, _ = get_dataset(func_type, n_samples=n_samples, noise=0.02)
    seq_train, _ = get_dataset(func_type, n_samples=n_samples, noise=0.02, sequence=True, seq_len=10)

    y_raw = np.array([p[1] for p in train_data])
    y_mean, y_std = y_raw.mean(), y_raw.std()
    norm = lambda y: (y - y_mean) / (y_std + 1e-7)

    train_data = [(x, norm(y)) for x, y in train_data]
    seq_train = [(x, norm(y)) for x, y in seq_train]

    x_test = np.linspace(-1, 1, 200).astype(np.float32).reshape(-1, 1)
    y_test_true = norm(np.array([p[1] for p in complex_function_generator(func_type, 200, noise=0, sorted_x=True)]).flatten())

    methods = [
        ('MLP', GenericModel(create_mlp(128), BL.MeanSquaredError()), False),
        ('LSTM_Complex', GenericModel(create_lstm_predictor(64, 32), BL.MeanSquaredError()), True),
        ('LSTM_Logical', GenericModel(create_lstm_predictor(64, 32),
                                      BL.CompositeLoss([(BL.MeanSquaredError(), 1.0), (BL.LogicalConstraintLoss('equivalent'), 1.0)]),
                                      BL.StateTransitionLoss(weight=0.05)), True),
    ]

    results = {}
    for name, model, use_seq in methods:
        print(f"Training {name}...")
        it = chainer.iterators.SerialIterator(seq_train if use_seq else train_data, 64)
        opt = chainer.optimizers.Adam(alpha=0.001)
        if 'Logical' in name:
            opt = LogicalOptimizer(opt)
            model.optimizer = opt
        opt.setup(model)

        trainer = training.Trainer(training.StandardUpdater(it, opt), (epochs, 'epoch'), out=f'_tmp_v4_{func_type}_{name}')
        if 'Logical' in name:
            trainer.extend(LogicalWeightScheduler(opt, factor=0.95, interval=100))
        trainer.run()

        if use_seq:
            for l in model.predictor:
                if hasattr(l, 'reset_state'): l.reset_state()
            preds = [model.predictor(chainer.Variable(xi.reshape(1,1))).data.flatten()[0] for xi in x_test]
            y_pred = np.array(preds)
        else:
            y_pred = model.predictor(chainer.Variable(x_test)).data.flatten()
        results[name] = y_pred

    poly_coeffs = np.polyfit(np.array([p[0] for p in train_data]).flatten(),
                             np.array([p[1] for p in train_data]).flatten(), 12)
    results['Poly_Deg12'] = np.polyval(poly_coeffs, x_test.flatten())

    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_test_true, 'k--', label='True', alpha=0.5)
    for name, yp in results.items():
        mse = np.mean((yp - y_test_true)**2)
        plt.plot(x_test, yp, label=f'{name} (MSE: {mse:.4f})')
    plt.title(f'Iteration V5: {func_type}')
    plt.legend(); plt.grid(True)
    path = f'eval_v5_{func_type}.png'
    plt.savefig(path); print(f"Saved {path}")
    import shutil
    shutil.rmtree(f'_tmp_v4_{func_type}_MLP', ignore_errors=True)
    shutil.rmtree(f'_tmp_v4_{func_type}_LSTM_Complex', ignore_errors=True)
    shutil.rmtree(f'_tmp_v4_{func_type}_LSTM_Logical', ignore_errors=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='all')
    parser.add_argument('--epochs', type=int, default=150)
    args = parser.parse_args()

    if args.target == 'all':
        targets = ['fixed_poly', 'fourier_series']
    else:
        targets = [args.target]

    for t in targets:
        train_and_evaluate(t, epochs=args.epochs)
