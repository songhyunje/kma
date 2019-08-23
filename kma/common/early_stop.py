class EarlyStopping:
    """Early stops the training if validation score doesn't improve after a given patience."""
    def __init__(self, min_epoch=40, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.min_epoch = min_epoch
        self.early_stop = False

    def __call__(self, epoch, score):
        if epoch < self.min_epoch:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self.best_score < score:
            self.counter += 1
            if self.patience and self.counter > self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
