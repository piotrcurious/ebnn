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
from ebnn.utils.energy_landscape import PotentialEnergy, InteractionEnergy, KineticEnergy, HamiltonianSystem
from ebnn.utils.structural_awareness import StructuralAwareness
from ebnn.utils.dreaming_engine import DreamingInference
from ebnn.utils.aware_optimizer import AwareHamiltonianOptimizer
from dataset_generator import get_dataset, complex_function_generator

class ExploratoryModel(chainer.Chain):
    def __init__(self, predictor, energy_system):
        super(ExploratoryModel, self).__init__()
        with self.init_scope():
            self.predictor = predictor
        self.energy_system = energy_system
        self.dreamer = DreamingInference(self.predictor, self.energy_system)

    def __call__(self, x, t):
        for name, link in self.predictor.namedlinks():
            if hasattr(link, 'reset_state'): link.reset_state()

        y = self.predictor(x)
        loss = self.energy_system(self.predictor, x, y, t)

        reporter.report({'loss': loss, 'mse': F.mean_squared_error(y, t)}, self)
        return loss

    def exploratory_inference(self, x, t_ref, steps=5):
        # Refine prediction via Dreaming
        refined_x = self.dreamer.dream(x, t_ref, steps=steps)
        return self.predictor(refined_x)

def train_exploratory_system(func_type, n_samples=1000, epochs=200):
    print(f"\n--- [Exploratory System] {func_type} ---")
    train_data_raw, _ = get_dataset(func_type, n_samples=n_samples, noise=0.01)

    y_vals = np.array([p[1] for p in train_data_raw])
    y_min, y_max = y_vals.min(), y_vals.max()
    scale = lambda y: (y - y_min) / (y_max - y_min + 1e-7)
    unscale = lambda y: y * (y_max - y_min) + y_min
    train_data = [(x, scale(y).astype(np.float32).reshape(1)) for x, y in train_data_raw]

    # Define Hamiltonian Energy Landscape
    energy_landscape = PotentialEnergy() + 0.001 * InteractionEnergy() + 0.001 * KineticEnergy()
    hamiltonian = HamiltonianSystem(energy_landscape)

    predictor = chainer.Sequential(
        BL.BinaryLSTM(1, 64),
        L.Linear(None, 32), F.relu,
        L.Linear(32, 1)
    )
    model = ExploratoryModel(predictor, hamiltonian)

    opt = AwareHamiltonianOptimizer(chainer.optimizers.Adam(alpha=0.0005))
    opt.setup(model)

    it = chainer.iterators.SerialIterator(train_data, 32)
    trainer = training.Trainer(training.StandardUpdater(it, opt), (epochs, 'epoch'), out=f'_tmp_exploratory_{func_type}')
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/mse']))
    trainer.run()

    # Evaluation with Awareness and Dreaming
    x_test = np.linspace(-1, 1, 100).astype(np.float32).reshape(-1, 1)
    y_test_true = np.array([p[1] for p in complex_function_generator(func_type, 100, noise=0, sorted_x=True)]).flatten()

    with chainer.using_config('train', False):
        for name, link in model.predictor.namedlinks():
            if hasattr(link, 'reset_state'): link.reset_state()
        # Standard Prediction
        y_pred_std = unscale(model.predictor(x_test).data.flatten())

        # Exploratory Inference (Dreaming)
        # We 'dream' about the test targets to see how the model generalizes its landscape
        for name, link in model.predictor.namedlinks():
            if hasattr(link, 'reset_state'): link.reset_state()
        x_var = chainer.Variable(x_test)
        t_dummy = chainer.Variable(scale(y_test_true).astype(np.float32).reshape(-1, 1))
        y_pred_dream = unscale(model.exploratory_inference(x_var, t_dummy, steps=10).data.flatten())

    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_test_true, 'k--', label='True', alpha=0.5)
    plt.plot(x_test, y_pred_std, label=f'Standard (MSE: {np.mean((y_pred_std - y_test_true)**2):.4f})')
    plt.plot(x_test, y_pred_dream, label=f'Dreamed (MSE: {np.mean((y_pred_dream - y_test_true)**2):.4f})')
    plt.title(f'Exploratory System: {func_type}')
    plt.legend(); plt.grid(True)
    path = f'exploratory_result_{func_type}.png'
    plt.savefig(path); print(f"Saved {path}")
    plt.close()

    # Log Awareness
    awareness = StructuralAwareness.map_connectivity(model.predictor)
    print(f"Final Structural Awareness: {list(awareness.keys())}")

if __name__ == '__main__':
    train_exploratory_system('fourier_series', epochs=100)
    train_exploratory_system('fixed_poly', epochs=100)
