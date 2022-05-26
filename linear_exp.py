from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
from utilities import generate_all_binaries, create_test_matrix_11
import random


# Hyperparameters:
dimension = 11
batch_size = 64
device = torch.device("cpu")
max_epochs = 10000  # This is the maximum number of epochs, the training will be stopped when training loss becomes lower than a threshold.
number_of_experiments = 20  # Number of times that the experiment will be repeated.
training_size = 1500  # Number of training samples
learning_rate = 0.00001
width = 256  # Width of the hidden layers


def calculate_stair_case(A):
    """
    A simple linear function: f(x_1, ..., x_11) = 1 + 2x_1 - 3x_2 + ... + 12x_11
    """
    return 1 + 2 * A[:, 0] - 3 * A[:, 1] + 4 * A[:, 2] - 5 * A[:, 3] + 6 * A[:, 4] - 7 * A[:, 5] + 8 * A[:, 6] - 9 * A[:, 7] + 10 * A[:, 8] - 11 * A[:, 9] + 12 * A[:, 10]


def loss_on_frozen_index(train_X, test_X, test_y, i, depth=3, alpha_init=0.5, verbose_acc=0, name=""):
    """
    This function freezes one index, trains the model, and calculates the generalization error.
    :param train_X: Training set (X)
    :param test_X: Test set (X)
    :param test_y: Test set (unfrozen y)
    :param i: frozen index
    :param depth: depth of the linear neural network
    :param alpha_init: initialization scale, each layer is initialized with n^(-alpha).
    :return: in-distribution generalization loss, out-of-distribution generalization loss, number of epochs
    """
    # Defining the model
    layers = [nn.Linear(dimension, width)]
    layers += [nn.Linear(width, width) for _ in range(depth - 2)]
    layers.append(nn.Linear(width, 1))
    model = torch.nn.Sequential(*layers).to(device)

    # Applying proper initialization
    def init_weights(layer):
        if isinstance(layer, nn.Linear):
            stdv = 1. / layer.weight.size(1) ** alpha_init
            torch.nn.init.uniform_(layer.weight, -stdv, stdv)
            torch.nn.init.uniform_(layer.bias, -stdv, stdv)
    model.apply(init_weights)

    # Creating and preparing the frozen dataset
    train_X_frozen = train_X.copy()
    test_X_frozen = test_X.copy()
    train_X_frozen[:, i] = 1
    test_X_frozen[:, i] = 1
    # Reshaping
    train_y_frozen = calculate_stair_case(train_X_frozen).reshape(-1, 1)
    test_y_frozen = calculate_stair_case(test_X_frozen).reshape(-1, 1)
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
    test_ds = TensorDataset(test_X, test_y)
    test_dl = DataLoader(test_ds, batch_size=512)
    test_frozen_ds = TensorDataset(test_X_frozen, test_y_frozen)
    test_frozen_dl = DataLoader(test_frozen_ds, batch_size=512)

    # Defining the optimizer
    opt = optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()

    for epoch in range(max_epochs):
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
        if (verbose_acc > 0 and epoch % verbose_acc == 0) or epoch == max_epochs - 1:
            with torch.no_grad():
                gen_loss = sum(loss_func(model(xb), yb) for xb, yb in test_dl)  # Error for both in and out of distribution samples
                in_dist_loss = sum(loss_func(model(xb), yb) for xb, yb in test_frozen_dl)  # Error for in distribution samples (frozen ones)
                print("Expected vale for generalization error given the Boolean influence:", (test_y - test_y_frozen).pow(2).mean())
                print(f"Model: {name}, Epoch: {epoch}, Train Loss: {train_loss / len(train_dl)}, In Distribution Loss: {in_dist_loss / len(test_frozen_dl)}, Generalization Loss: {gen_loss / len(test_dl)}")

            if (train_loss / len(train_dl)) < 10 ** (-8) and epoch >= 49:
                break
    return in_dist_loss.cpu().detach().numpy() / len(test_frozen_dl), gen_loss.cpu().detach().numpy() / len(test_dl), epoch


def wrapper(input_tuple):
    """
    This is just a wrapper function for loss_on_frozen_index function
    """
    train_X, test_X, test_y, index, exp_iter, depth, alpha_init = input_tuple
    random_seed = exp_iter * 1000 + index
    torch.manual_seed(random_seed)
    return loss_on_frozen_index(train_X, test_X, test_y, index, verbose_acc=10, name=f"Exp={exp_iter}, Frozen={index}",
                                depth=depth, alpha_init=alpha_init)


if __name__ == '__main__':
    parser = ArgumentParser(description="Training for linear neural networks",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-depth', default=3, type=int, help='depth of the neural network')
    parser.add_argument('-alpha-init', default=0.5, type=float, help='alpha to determine the scale of initialization')
    args = parser.parse_args()

    results_total = []
    for exp_iter in range(number_of_experiments):
        results_current_run = []
        np.random.seed(exp_iter * 1000)
        random.seed(exp_iter * 1000)
        train_X = create_test_matrix_11(training_size, dimension)
        test_X = generate_all_binaries(dimension)  # If dimension is small, we can use all binary sequences for testing.
        test_y = calculate_stair_case(test_X)
        for i in range(dimension):
            print(f"# Results for frozen index = {i}:")
            results = [0.0, 0.0, 0.0]
            results[0], results[1], results[2] = wrapper((train_X.copy(), test_X.copy(), test_y.copy(), i, exp_iter, args.depth, args.alpha_init))
            # results[0] is the in-distribution loss and shows how good the training was.
            # results[1] shows the generalization error to the unfrozen data.
            results_current_run.append(results)
        results_total.append(results_current_run)
        print(np.array(results_total).mean(axis=0))
        print("Std of ood generalization error per index:", np.array(results_total).std(axis=0)[:, 1])
        saved_data = {'results_training': results_total, 'end_index': dimension}
        with open(f"linear_function_{args.depth}_{args.alpha_init}.npz", "wb") as f:
            np.savez(f, **saved_data)
