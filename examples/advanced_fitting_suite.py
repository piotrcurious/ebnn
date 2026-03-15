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
from dataset_generator import get_dataset

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

        # Check if x is a sequence (batch_size, seq_len, in_dim)
        if x.ndim == 3:
            # Sequence training
            loss = 0
            seq_len = x.shape[1]
            for i in range(seq_len):
                xi = x[:, i, :]
                ti = t[:, i, :]
                yi = self.predictor(xi)

                if isinstance(self.loss_func, BL.CompositeLoss):
                    # Sum all losses with their weights, potentially scaling logical ones
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
                    loss += logical_weight * self.state_loss(self, xi, ti)
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

def train_and_evaluate(func_type, n_samples=1000, epochs=150):
    print(f"\n--- Evaluating Function Type: {func_type} ---")

    # Base datasets
    train_data, test_data = get_dataset(func_type, n_samples=n_samples, noise=0.02)

    # Normalize targets to [0, 1] for better logical loss compatibility
    y_train_raw = np.array([p[1] for p in train_data])
    y_min, y_max = y_train_raw.min(), y_train_raw.max()

    def normalize(data, ymin, ymax):
        return [(x, (y - ymin) / (ymax - ymin + 1e-7)) for x, y in data]

    train_data = normalize(train_data, y_min, y_max)
    test_data = normalize(test_data, y_min, y_max)

    # Sequence datasets for LSTM
    seq_train, seq_test = get_dataset(func_type, n_samples=n_samples, noise=0.02, sequence=True, seq_len=5)
    seq_train = [(x, (y - y_min) / (y_max - y_min + 1e-7)) for x, y in seq_train]

    x_test_full = np.linspace(-1, 1, 200).astype(np.float32).reshape(-1, 1)
    # Get true y for testing (no noise)
    from dataset_generator import complex_function_generator
    y_test_true_raw = np.array([p[1] for p in complex_function_generator(func_type, 200, noise=0, sorted_x=True)]).flatten()
    y_test_true = (y_test_true_raw - y_min) / (y_max - y_min + 1e-7)

    methods = [
        ('NN_MSE', GenericModel(create_mlp(64), BL.MeanSquaredError()), False),
        ('LSTM_Basic', GenericModel(create_lstm_predictor(32, 16), BL.MeanSquaredError()), True),
        ('LSTM_Logical', GenericModel(create_lstm_predictor(32, 16),
                                      BL.CompositeLoss([(BL.MeanSquaredError(), 1.0), (BL.LogicalConstraintLoss('equivalent'), 1.0)]),
                                      BL.StateTransitionLoss(weight=0.1)), True),
    ]

    results = {}

    for name, model, use_seq in methods:
        print(f"Training {name}...")
        data = seq_train if use_seq else train_data
        train_iter = chainer.iterators.SerialIterator(data, 32)

        base_opt = chainer.optimizers.Adam(alpha=0.001)
        if name == 'LSTM_Logical':
            optimizer = LogicalOptimizer(base_opt)
            optimizer.setup(model)
            model.optimizer = optimizer
        else:
            optimizer = base_opt
            optimizer.setup(model)

        updater = training.StandardUpdater(train_iter, optimizer)
        trainer = training.Trainer(updater, (epochs, 'epoch'), out=f'_tmp_{func_type}_{name}')

        if name == 'LSTM_Logical':
            trainer.extend(LogicalWeightScheduler(optimizer, factor=0.98, interval=50))

        trainer.run()

        # Evaluation
        if use_seq:
            for link in model.predictor:
                if hasattr(link, 'reset_state'):
                    link.reset_state()
            preds = []
            for xi in x_test_full:
                xv = chainer.Variable(xi.reshape(1, 1))
                yv = model.predictor(xv)
                preds.append(yv.data.flatten()[0])
            y_pred = np.array(preds)
        else:
            x_test_var = chainer.Variable(x_test_full)
            y_pred = model.predictor(x_test_var).data.flatten()

        results[name] = y_pred

    # Classical Polynomial Fit (Degree 7)
    x_train_flat = np.array([p[0] for p in train_data]).flatten()
    y_train_flat = np.array([p[1] for p in train_data]).flatten()
    poly_coeffs = np.polyfit(x_train_flat, y_train_flat, 7)
    y_poly_pred = np.polyval(poly_coeffs, x_test_full.flatten())
    results['Classical_Poly7'] = y_poly_pred

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_test_full, y_test_true, label='True Function (norm)', color='black', linestyle='--', alpha=0.6)

    for name, y_pred in results.items():
        mse = np.mean((y_pred - y_test_true)**2)
        plt.plot(x_test_full, y_pred, label=f'{name} (MSE: {mse:.4f})', linewidth=1.5)

    plt.title(f'Advanced Fitting comparison for {func_type}')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y (normalized)')
    plot_path = f'eval_v3_{func_type}.png'
    plt.savefig(plot_path)
    print(f"Graph saved as {plot_path}")

    import shutil
    for name, _, _ in methods:
        shutil.rmtree(f'_tmp_{func_type}_{name}', ignore_errors=True)

if __name__ == '__main__':
    for ft in ['compound_sine', 'sine_exp_decay', 'varying_noise', 'polynomial']:
        train_and_evaluate(ft, epochs=100)
