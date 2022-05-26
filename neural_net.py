from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import random
import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
from utilities import create_test_matrix_11, generate_all_binaries, calculate_Boolean_influence, calculate_stair_case, f1, f3, calculate_fourier_coefficients, create_monomials_mask, calc_PVR_function_cyclic

import token_transformer
import mixer
import models


# Note: In the code we have used l2 loss. While, in the paper we define the generalization loss as half of the l2 loss.
# General setup of the experiments
dimension = 11  # Dimension of the input (11 for 3 pointer bits, 20 for 4 pointer bits and 14 for staircase)
pointer_bits = 3


exp_arch = 'mlp'  # Model selected from 'mlp', 'transformer', 'mixer'
function_name = 'f1'  # f3 (PVR with 3 pointer bits, w=3, and majority), f1, staircase, or PVR
batch_size = 64 if dimension <= 15 else 1024
test_batch_size = 1024
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 150
# staircase function: 150 epochs, PVRs with 3 pointer bits: 100 epochs, f1: 100 epochs
# PVR with 4 pointer bits and parity: 80 epochs, PVR with 4 pointer bits and majority/min: 30 epochs
training_size = 1900 if dimension == 11 else (10000 if dimension == 14 else 100000)
# Number of training sample, for PVR with 4 pointer bits (20 in total), we use 100000 training samples.
number_of_experiments = 40
window_size = 1
agg_dict = {'parity': np.prod, 'majority': lambda x: np.sign(np.sum(x)), 'min': np.min, 'max': np.max}
agg = agg_dict['parity']


momentum = 0.9
learning_rate = 0.00005
# When using Adam, we use 0.00005 as the learning rate.
# When using SGD, for staircase function we use 0.0005 for Transformer and MLP-Mixer and 0.0025 for MLP.
# For f1, we use 0.0001 and for the PVR example (3 pointer bits, w=3, majority aka f3) we use 0.001


def PVR(X):
    return calc_PVR_function_cyclic(X, pointer_bits, window_size, agg)


target_function = {'staircase': calculate_stair_case, 'f1': f1, 'f3': f3, 'PVR': PVR}[function_name]
# This is the target function that the model is going to learn.
start_index = 7  # We may not want to test all indices.
end_index = 7


def build_model(arch, dimension):
    if arch == 'mlp':
        model = models.MLP(input_dimension=dimension)
    elif arch == 'transformer':
        model = token_transformer.TokenTransformer(
                seq_len=dimension, output_dim=1, dim=256, depth=12, heads=6, mlp_dim=256)
    elif arch == 'mixer':
        model = mixer.TokenMixer(
                seq_len=dimension, output_dim=1, dim=256, depth=12)
    return model.to(device)


def loss_on_frozen_index(train_X, test_X, test_y, i, verbose_acc=0, name="", monomials=None):
    """
    For a given index, it will freeze the training set. Train the model and calculates its generalization error.
    :param train_X: The training set X.
    :param test_X: The test set X.
    :param test_y: The test set y.
    :param i: The index which is going to be frozen.
    :param verbose_acc: Verbose for acc of model. For example if set to 10, the loss of model will be printed after
    each 10 epochs.
    :param verbose_weight: Verbose for weights. If non-zero, if will print the weights of the frozen bit in the
    first layer and also biases of the first layer. Note that when we freeze a bit it acts like a bias so for example
    the difference of aforementioned weights and biases will remain constant.
    :param name: Name of the model.
    :return: The function returns (model, test_error, generalization error). By test error we mean the error on
    the frozen test set and by generalization error we mean the error for test set when it's not frozen, i.e.,
    generalization error of in and out of distribution samples.
    """
    model = build_model(exp_arch, dimension)
    coefficients = []

    # Creating and preparing the frozen dataset
    train_X_frozen = train_X.copy()
    test_X_frozen = test_X.copy()
    train_X_frozen[:, i] = 1
    test_X_frozen[:, i] = 1
    # Reshaping
    train_y_frozen = target_function(train_X_frozen).reshape(-1, 1)
    test_y_frozen = target_function(test_X_frozen).reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    # Creating pytorch tensors
    train_X_frozen = torch.tensor(train_X_frozen, device=device)
    train_y_frozen = torch.tensor(train_y_frozen, device=device)
    test_X_frozen = torch.tensor(test_X_frozen, device=device)
    test_y_frozen = torch.tensor(test_y_frozen, device=device)
    test_X = torch.tensor(test_X, device=device)
    test_y = torch.tensor(test_y, device=device)

    # Creating datasets and dataloaders
    train_ds = TensorDataset(train_X_frozen, train_y_frozen)
    train_dl = DataLoader(train_ds, batch_size=batch_size)
    test_frozen_ds = TensorDataset(test_X_frozen, test_y_frozen)
    test_frozen_dl = DataLoader(test_frozen_ds, batch_size=test_batch_size)
    test_ds = TensorDataset(test_X, test_y)
    test_dl = DataLoader(test_ds, batch_size=test_batch_size)

    # Defining the optimizer
    # opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)
            train_loss += loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        model.eval()
        with torch.no_grad():
            if monomials is not None:
                coefficients.append(calculate_fourier_coefficients(monomials, test_X.cpu().detach().numpy(),
                                                               model(test_X).cpu().detach().numpy()))
            if (verbose_acc > 0 and epoch % verbose_acc == 0) or epoch == epochs - 1:
                gen_loss = sum(loss_func(model(xb), yb) for xb, yb in test_dl)  # Error for both in and out of distribution samples
                in_dist_loss = sum(loss_func(model(xb), yb) for xb, yb in test_frozen_dl)  # Error for in distribution samples (frozen ones)
                print("Expected value for generalization error given the Boolean influence:", (test_y - test_y_frozen).pow(2).mean())
                print(f"Model: {name}, Epoch: {epoch}, Train Loss: {train_loss / len(train_dl)}, In Distribution Loss: {in_dist_loss / len(test_frozen_dl)}, Generalization Loss: {gen_loss / len(test_dl)}")  # Test loss stands for frozen samples and Dist loss stands for unfrozen samples.
    return in_dist_loss.cpu().detach().numpy() / len(test_frozen_dl), gen_loss.cpu().detach().numpy() / len(test_dl), coefficients


def wrapper(input_tuple):
    """
    This is just a wrapper function for loss_on_frozen_index function
    """
    train_X, test_X, test_y, index, exp_iter = input_tuple
    random_seed = exp_iter * 1000 + index
    torch.manual_seed(random_seed)
    return loss_on_frozen_index(train_X, test_X, test_y, index, verbose_acc=1, name=f"Exp={exp_iter}, Frozen={index}",
                                monomials=create_monomials_mask(function_name, dimension, index))


if __name__ == '__main__':
    parser = ArgumentParser(description="Training script for neural networks on different functions",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-func', default='f3', type=str, help='name of the function')
    parser.add_argument('-model', default='transformer', type=str, help='name of the model')
    parser.add_argument('-agg', default='parity', type=str, help='name of the aggregation function in case of using PVR')
    parser.add_argument('-w', default=3, type=int, help='number of window bits in case of using PVR')
    parser.add_argument('-epochs', default=100, type=int, help='number of epochs')
    args = parser.parse_args()

    window_size = args.w
    agg = agg_dict[args.agg]
    exp_arch = args.model
    epochs = args.epochs
    target_function = {'staircase': calculate_stair_case, 'f1': f1, 'f3': f3, 'PVR': PVR}[args.func]

    results_total = []
    monomials_total = [[] for _ in range(dimension)]
    for exp_iter in range(number_of_experiments):
        results_current_run = []
        np.random.seed(exp_iter * 1000)
        random.seed(exp_iter * 1000)
        train_X = create_test_matrix_11(training_size, dimension)
        test_X = generate_all_binaries(dimension)  # If dimension is small, we can use all binaries for testing.
        test_y = target_function(test_X)
        for i in range(start_index, end_index + 1):
            print(f"# Results for frozen index = {i}:")
            results = [0.0, 0.0]
            results[0], results[1], index_monomials = wrapper((train_X.copy(), test_X.copy(), test_y.copy(), i, exp_iter))
            # results[0] is the in-distribution loss and shows how good the training was.
            # results[1] shows the generalization error when test set is not frozen.
            results_current_run.append(results)
            monomials_total[i].append(index_monomials)
        results_total.append(results_current_run)
        print(np.array(results_total).mean(axis=0))
        print("Std of ood generalization error per index:", np.array(results_total).std(axis=0)[:, 1])
        saved_data = {'results_training': results_total, 'start_index': start_index, 'end_index': dimension}
        for i in range(start_index, dimension):
            saved_data[f"mask_{i}"] = create_monomials_mask(function_name, dimension, i)
            saved_data[f"coef_{i}"] = np.array(monomials_total[i])
        with open(f"{function_name}_{exp_arch}_{args.agg}_{args.w}.npz", "wb") as f:
            np.savez(f, **saved_data)
    print("Expected result based on Boolean influence:", calculate_Boolean_influence(target_function, dimension)[start_index:] * 2)
