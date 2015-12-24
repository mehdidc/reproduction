from invoke import task
from model import ConvModel, Bogdan


@task
def train(kind="dense"):
    from lasagnekit.datasets.mnist import MNIST
    from sklearn.utils import shuffle
    from keras.utils import np_utils
    from keras.callbacks import ModelCheckpoint
    from sklearn.cross_validation import train_test_split
    import numpy as np

    from hp_toolkit.hp import (minimize_fn_with_hyperopt, find_all_hp)
    from hp_toolkit.hp import instantiate_default_model

    from helpers import LearningRateScheduler
    np.random.seed(2)

    data = MNIST()
    data.load()
    img_dim = data.img_dim
    if len(img_dim) == 2:
        img_dim = [1] + list(img_dim)

    X, y = data.X, data.y
    y = np_utils.to_categorical(y, data.output_dim)
    X, y = shuffle(X, y)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.14286)
    X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, test_size=0.1667)

    def init_func(model, **opt):
        model.build(nb_features=X.shape[1], nb_outputs=data.output_dim,
                    imshape=img_dim)
        if "X_valid" in opt and "y_valid" in opt:
            model.validation_data = (opt["X_valid"], opt["y_valid"])
        return model

    def classification_error(model, X, y):
        return (model.predict(X) != y.argmax(axis=1)).mean()
    model = instantiate_default_model(Bogdan)
    model = init_func(model)

    def schedule(epoch, lr):
        return np.array(lr * 0.98, dtype="float32")
    checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5",
                                   monitor="val_acc",
                                   verbose=1, save_best_only=True)
    learning_rate_scheduler = LearningRateScheduler(schedule)
    model.fit(X_train, y_train,
              callbacks=[learning_rate_scheduler, checkpointer])
    model.model.load_weights("/tmp/weights.hdf5")
    print((model.predict(X_test) == y_test.argmax(axis=1)).mean())
    return

    all_hp, all_scores = find_all_hp(
            ConvModel,
            minimize_fn_with_hyperopt,
            X_train, X_valid,
            y_train, y_valid,
            eval_function=classification_error,
            max_evaluations=20,
            model_init_func=init_func)
    print(min(all_scores))
