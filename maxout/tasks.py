from invoke import task
from model import DenseModel


@task
def train(kind="dense"):
    from lasagnekit.datasets.mnist import MNIST
    from sklearn.utils import shuffle
    from keras.utils import np_utils
    from sklearn.cross_validation import train_test_split
    import numpy as np

    from hp_toolkit.hp import (minimize_fn_with_hyperopt, find_all_hp)
    np.random.seed(2)

    data = MNIST()
    data.load()

    X, y = data.X, data.y
    y = np_utils.to_categorical(y, data.output_dim)
    X, y = shuffle(X, y)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.15)
    X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, test_size=0.15)

    def init_func(model, **opt):
        model.build(nb_features=X.shape[1], nb_outputs=data.output_dim)
        model.validation_data = (opt["X_valid"], opt["y_valid"])
        return model

    def classification_error(model, X, y):
        return (model.predict(X) != y.argmax(axis=1)).mean()

    all_hp, all_scores = find_all_hp(
            DenseModel,
            minimize_fn_with_hyperopt,
            X_train, X_valid,
            y_train, y_valid,
            eval_function=classification_error,
            max_evaluations=100,
            model_init_func=init_func)
    print(min(all_scores))
