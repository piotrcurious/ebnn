import numpy as np

def complex_function_generator(func_type='compound_sine', n_samples=5000, noise=0.05, sorted_x=False):
    """
    Enhanced generator with more diverse function types and higher sample counts.
    """
    if sorted_x:
        x = np.linspace(-1, 1, n_samples).astype(np.float32)
    else:
        x = np.random.uniform(-1, 1, n_samples).astype(np.float32)

    if func_type == 'compound_sine':
        y = np.sin(5 * x) + 0.5 * np.sin(15 * x) + 0.2 * np.cos(30 * x)
    elif func_type == 'sine_exp_decay':
        y = np.sin(12 * x) * np.exp(-1.5 * np.abs(x))
    elif func_type == 'varying_noise':
        y = np.sin(5 * x) * np.cos(3 * x)
        local_noise = noise * (1 + 2 * np.abs(x))
        y += np.random.normal(0, local_noise, n_samples)
    elif func_type == 'fixed_poly':
        # Fixed 7th degree polynomial for stable benchmarking
        coeffs = np.array([1.5, -2.0, 0.5, 3.0, -1.2, 0.8, -0.5, 0.2], dtype=np.float32)
        y = np.polyval(coeffs, x)
    elif func_type == 'step_periodic':
        y = np.sign(np.sin(5 * x)) * 0.5 + 0.5 * np.sin(10 * x)
    elif func_type == 'fourier_series':
        y = sum([(1.0/n) * np.sin(n * np.pi * x) for n in range(1, 6)])
    else:
        # Default complex mix
        y = np.sin(10 * x) * np.cos(5 * x)

    if func_type != 'varying_noise':
        y += np.random.normal(0, noise, n_samples)

    y = y.astype(np.float32)

    for i in range(n_samples):
        yield x[i:i+1], y[i:i+1]

def get_dataset(func_type='compound_sine', n_samples=5000, noise=0.05, train_ratio=0.8, sequence=False, seq_len=20):
    if sequence:
        # For LSTMs, more sequence length can help capture longer dependencies
        data_points = list(complex_function_generator(func_type, n_samples, noise, sorted_x=True))
        sequences = []
        for i in range(len(data_points) - seq_len):
            seq = data_points[i : i + seq_len]
            seq_x = np.stack([p[0] for p in seq]).astype(np.float32)
            seq_y = np.stack([p[1] for p in seq]).astype(np.float32)
            sequences.append((seq_x, seq_y))

        np.random.shuffle(sequences)
        n_train = int(len(sequences) * train_ratio)
        return sequences[:n_train], sequences[n_train:]
    else:
        gen = list(complex_function_generator(func_type, n_samples, noise))
        np.random.shuffle(gen)
        n_train = int(n_samples * train_ratio)
        return gen[:n_train], gen[n_train:]
