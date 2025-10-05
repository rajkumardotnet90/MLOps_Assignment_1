def train_kernel_ridge(X_train, y_train, alpha=1.0, kernel='linear'):
    from sklearn.kernel_ridge import KernelRidge
    
    model = KernelRidge(alpha=alpha, kernel=kernel)
    model.fit(X_train, y_train)
    
    return model