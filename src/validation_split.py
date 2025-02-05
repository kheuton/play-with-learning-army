

def data_split(X, y, val_frac, test_frac):
    assert val_frac + test_frac < 1
    val_size = int(len(X) * val_frac)
    test_size = int(len(X) * test_frac)
    X_val, y_val = X[:val_size], y[:val_size]
    X_test, y_test = X[val_size:val_size + test_size], y[val_size:val_size + test_size]
    X_train, y_train = X[val_size + test_size:], y[val_size + test_size:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
