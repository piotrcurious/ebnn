import numpy as np

def complex_function_generator(func_type='compound_sine', n_samples=2000, noise=0.1, sorted_x=False):
    if sorted_x:
        x = np.linspace(-1, 1, n_samples).astype(np.float32)
    else:
        x = np.random.uniform(-1, 1, n_samples).astype(np.float32)

    if func_type == 'compound_sine':
        y = np.sin(5 * x) + 0.5 * np.sin(15 * x)
    elif func_type == 'sine_exp_decay':
        y = np.sin(10 * x) * np.exp(-np.abs(x))
    elif func_type == 'varying_noise':
        y = np.sin(5 * x)
        local_noise = noise * (1 + np.abs(x))
        y += np.random.normal(0, local_noise, n_samples)
    elif func_type == 'polynomial':
        coeffs = [0.5, -0.2, 0.8, -0.5] # 0.5 - 0.2x + 0.8x^2 - 0.5x^3
        y = np.polyval(coeffs[::-1], x)
    else:
        y = np.sin(x)

    if func_type != 'varying_noise':
        y += np.random.normal(0, noise, n_samples)

    y = y.astype(np.float32)

    for i in range(n_samples):
        yield x[i:i+1], y[i:i+1]

def get_dataset(func_type='compound_sine', n_samples=2000, noise=0.1, train_ratio=0.8, sequence=False, seq_len=10):
    if sequence:
        # Use sorted X for sequences to have temporal meaning
        data_points = list(complex_function_generator(func_type, n_samples, noise, sorted_x=True))
        sequences = []
        for i in range(len(data_points) - seq_len):
            seq = data_points[i : i + seq_len]
            # LSTM typically wants (seq_len, feature_dim)
            seq_x = np.stack([p[0] for p in seq]).astype(np.float32)
            seq_y = np.stack([p[1] for p in seq]).astype(np.float32)
            sequences.append((seq_x, seq_y))

        np.random.shuffle(sequences)
        n_train = int(len(sequences) * train_ratio)
        return sequences[:n_train], sequences[n_train:]
    else:
        gen = list(complex_function_generator(func_type, n_samples, noise))
        n_train = int(n_samples * train_ratio)
        return gen[:n_train], gen[n_train:]
