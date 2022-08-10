#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
import numpy as np
import pandas as pd

import gpflow
import tensorflow as tf
from gpflow.ci_utils import ci_niter

from tqdm import tqdm

from itertools import product


def run_adam(model, train_dataset, minibatch_size, iterations):
    """
    Utility function running the Adam optimizer
    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    with tqdm(total=iterations) as pbar:
        for step in range(iterations):
            optimization_step()
            pbar.update(1)
            if step % 10 == 0:
                elbo = -training_loss().numpy()
                # print(step, elbo)
                logf.append(elbo)
                pbar.desc = f"Elbo:{elbo}"
    return logf


class SVGP:
    def __init__(self, X=None, num_inducing=100, name=None):
        # just load up if there is a file to load
        if name is not None:
            self.load(name=name)
            return

        Y = np.array([1]*len(X), dtype=float).reshape(-1,1)

        # create a grid around the training data
        maxx, maxy, maxz = np.max(X, axis=0)
        minx, miny, minz = np.min(X, axis=0)
        n = 10
        extra = 0.15
        xs = np.linspace(minx-extra, maxx+extra, n)
        ys = np.linspace(miny-extra, maxy+extra, n)
        zs = np.linspace(minz-extra, maxz+extra, n)
        prod = list(product(xs, ys, zs))
        volume = np.array(prod)
        # and fill that space with 0 labels
        volume_Y = np.array([0]*len(volume), dtype=float).reshape(-1,1)

        # stack the training data and the volume around it up
        X = np.vstack([X, volume])
        Y = np.vstack([Y, volume_Y])


        # choose random points from the dataset for initial inducing points
        rand_inds = np.random.randint(0, len(X), num_inducing)
        Z = X[rand_inds, :].copy()

        # Create SVGP model.
        # kernel = gpflow.kernels.SquaredExponential(lengthscales=0.003)
        kernel = gpflow.kernels.Matern12(variance=0.01, lengthscales=0.0015)
        # whiten=True toggles the fₛ=Lu parameterization.
        # whiten=False uses the original parameterization.
        m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=len(X), whiten=True)
        # m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=len(X), whiten=False)
        # Enable the model to train the inducing locations.
        gpflow.set_trainable(m.inducing_variable, True)

        self.model = m
        self.X = X
        self.Y = Y
        self.Z = Z


    def train(self, num_iters=30000):
        # train
        minibatch_size = 100
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y)).repeat().shuffle(len(X))
        # Specify the number of optimization steps.
        maxiter = ci_niter(num_iters)
        # Perform optimization.
        logf = run_adam(self.model, train_dataset, minibatch_size, maxiter)
        self.model.predict_f_compiled = tf.function(self.model.predict_f, input_signature=[tf.TensorSpec(shape=[None, 3], dtype=tf.float64)])
        self.Z = self.model.inducing_variable.Z.numpy()


    def predict(self, pX):
        pX = np.atleast_2d(pX)
        pY, pYv = self.model.predict_f_compiled(pX[:,:3])
        pY = pY.numpy()[:,0]
        pYv = pYv.numpy()[:,0]
        return pY, pYv




    def save(self, name="svgp_model.tf_saved_model"):
        tf.saved_model.save(self.model, name)

        train = pd.DataFrame(data=np.hstack([self.X, self.Y]), columns=['x', 'y', 'z', 'label'])
        train.to_csv(f'{name}/train.csv', index=False)

        inducing = pd.DataFrame(data=self.Z, columns=['x', 'y', 'z'])
        inducing.to_csv(f'{name}/inducing.csv', index=False)


    def load(self, name="svgp_model.tf_saved_model"):
        self.model = tf.saved_model.load(name)

        train = pd.read_csv(f'{name}/train.csv')
        allX = train[['x','y','z','label']].to_numpy()
        self.X = allX[:,:3]
        self.Y = allX[:,3]

        inducing = pd.read_csv(f'{name}/inducing.csv')
        self.Z = inducing[['x', 'y', 'z']].to_numpy()



class MultiSVGP:
    def __init__(self, name_root='svgp_model_man', num_models=9):
        print("MultiSVGP Loading models")
        self.models = [SVGP(name=f'{name_root}{i}') for i in range(num_models)]

    def predict(self, pX):
        predictions = np.array([m.predict(pX) for m in self.models])
        most_conf = np.argmax(predictions[:,0,:], axis=0)
        return np.choose(most_conf, predictions)

    def predict_maneuver(self, pX, limit=0.95, man_id=None):
        # just get the pYs
        if man_id is None:
            predictions = np.array([m.predict(pX)[0] for m in self.models]).T
            # predictions is (N,9) now

            # find the most confident one and see if it is above the limit in its prediction
            most_conf = np.atleast_2d(np.argmax(predictions, axis=1)).T
            # most_conf is (N,1)
            pred_of_most_conf = np.take_along_axis(predictions, most_conf, axis=1)[:,0]
            # pred_of_most_conf is (N,)
            # get a mask for the ones that are above the limit
            above_limit = pred_of_most_conf >= limit
            res = np.array([9]*len(pX))
            # set the ones that are above limit as the pedicted maneuver
            # the rest are left at 9, labeled as "none"
            res[above_limit] = most_conf[above_limit,0]
            return res
        else:
            # if man_id is given, we just ask that maneuvers svgp
            # to classify it as "in" or "out"
            # instead of asking everyone about it
            res = np.array([9]*len(pX))
            for model_id, model in enumerate(self.models):
                mask = model_id == man_id
                pY = model.predict(pX[mask])[0]
                above_limit = pY >= limit
                # because you cant just do
                # res[mask][above_limit] = model_id
                # because memory shenanigans (it wont work)
                b = res[mask]
                b[above_limit] = model_id
                res[mask] = b

            return res


    @property
    def X(self):
        return np.vstack([m.X for m in self.models])

    @property
    def Y(self):
        return np.hstack([m.Y for m in self.models])

    @property
    def Z(self):
        return np.vstack([m.Z for m in self.models])


if __name__ == "__main__":
    from plot_data_json import Dataset
    from sim_to_real import Realifier
    import pickle

    np.set_printoptions(precision=4, floatmode='fixed', suppress=True)

    # self = MultiSVGP()
    # pX = np.random.random(size=(100,3))
    # predictions = np.array([m.predict(pX) for m in self.models])
    # most_conf = np.argmax(predictions[:,0,:], axis=0)



    # load up on data
    with open("realifier.pickle", 'rb+') as f:
        realifier = pickle.load(f)

    sim = Dataset('normal_9_500rpm_data.json', normalize_u=True, keep_underwater_only=True)
    # set the xformed points from the realifier into the sim dataset
    assert sim.xyz.shape == realifier.sim_xformed.shape
    sim.xdot[:,:3] = realifier.sim_xformed

    # train and save for each maneuver
    for man_id in range(9):
        print(f"Running for man {man_id}")
        x, xdot, u, t, accels = sim.get_maneuver(man_id)
        X = xdot[:,:3]

        svgp = SVGP(X, num_inducing=100)
        svgp.train(num_iters=30000)

        # save the model
        svgp.save(name=f"svgp_model_man{man_id}")









