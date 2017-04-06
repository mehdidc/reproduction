from invoke import task
from model import ConvModel, Bogdan


@task
def train():
    from lasagnekit.datasets.mnist import MNIST
    from sklearn.utils import shuffle
    from keras.utils import np_utils
    from keras.callbacks import ModelCheckpoint
    from sklearn.cross_validation import train_test_split
    import numpy as np

    from hp_toolkit.hp import (minimize_fn_with_hyperopt, find_all_hp)
    from hp_toolkit.hp import instantiate_default_model, instantiate_random_model

    from helpers import LearningRateScheduler

    from lightexperiments.light import Light
    import uuid

    seed = 13442
    np.random.seed(seed)

    light = Light()
    light.launch()
    light.initials()
    light.set_seed(seed)
    light.tag("empiricalml")

    dataset = MNIST
    light.set("dataset", dataset.__class__.__name__)

    data = dataset()
    data.load()
    data.X = data.X.reshape((data.X.shape[0], 1, 28, 28))
    img_dim = data.img_dim
    if len(img_dim) == 2:
        img_dim = [1] + list(img_dim)

    X, y = data.X, data.y
    y = np_utils.to_categorical(y, data.output_dim)
    X, y = shuffle(X, y)

    test_size = 0.15
    X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size)
    light.set("train_size", len(X_train_full))
    light.set("test_size", len(X_test))
    # X_train, X_valid, y_train, y_valid = train_test_split(
    #        X_train_full, y_train_full, test_size=0.1667)

    def init_func(model, **opt):
        model.build(nb_features=X.shape[1], nb_outputs=data.output_dim,
                    imshape=img_dim)
        if "X_valid" in opt and "y_valid" in opt:
            model.validation_data = (opt["X_valid"], opt["y_valid"])
        return model

    def classification_error(model, X, y):
        return (model.predict(X) != y.argmax(axis=1)).mean()

    model_cls = Bogdan
    params = model_cls.params.keys()

    model = instantiate_default_model(model_cls)
    model = init_func(model)

    for param in params:
        light.set(param, getattr(model, param))
        print("{}={}".format(param, getattr(model, param)))

    def schedule(epoch, lr):
        return np.array(lr * model.decay, dtype="float32")

    name = uuid.uuid4()
    with open("{}.json".format(name), "w") as fd:
        fd.write(model.model.to_json())

    filename = "{}.hdf5".format(name)
    checkpointer = ModelCheckpoint(filepath=filename,
                                   monitor="val_acc",
                                   verbose=1, save_best_only=True)
    learning_rate_scheduler = LearningRateScheduler(schedule)
    hist = model.fit(X_train_full, y_train_full,
                     callbacks=[learning_rate_scheduler, checkpointer])
    for k, v in hist.history.items():
        light.set(k, v)
    model.model.save_weights(filename)
    #model.model.load_weights(filename)
    test_error = classification_error(model, X_test, y_test)
    print(test_error)
    light.set("test_error", test_error)
    light.endings()
    light.store_experiment()
    light.close()
