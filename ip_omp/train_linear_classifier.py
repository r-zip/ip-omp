import argparse

import numpy as np
import torch
import tqdm

torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sparsity_level", type=int)  # positional argument


def compute_gradients():
    total_norm = 0
    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        total_norm += p.grad.data.norm(2).item()

    return total_norm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        curr_lr = param_group["lr"]
        return curr_lr


# build custom module for logistic regression
class LogisticRegression(torch.nn.Module):
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.bn_1 = torch.nn.BatchNorm1d(n_inputs)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = torch.nn.Linear(n_inputs, n_outputs)

    # make predictions
    def forward(self, x):
        y_pred = self.linear(self.dropout(x))
        return y_pred


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(dataset_name):
    if dataset_name == "cifar10":
        return LogisticRegression(n_inputs=128, n_outputs=10)

    elif dataset_name == "cifar100":
        return LogisticRegression(n_inputs=824, n_outputs=100)

    elif dataset_name == "cub":
        return LogisticRegression(n_inputs=208, n_outputs=200)

    elif dataset_name == "places365":
        return LogisticRegression(n_inputs=2207, n_outputs=365)

    elif dataset_name == "imagenet":
        return LogisticRegression(n_inputs=4523, n_outputs=1000)


def train(dataset, bs):
    batch_size = bs
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model.train()
    grad_norm = 0
    for data in tqdm.tqdm(dataloader):
        x, y = data

        x = x.to(device)

        y = y.to(device).long()

        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()

        grad_norm += compute_gradients() * (x.size(0) / len(dataset))

        optimizer.step()

    return loss.item()


def test(dataset, bs):
    batch_size = 1024
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model.eval()
    correct = 0
    for data in tqdm.tqdm(dataloader):
        x, y = data

        x = x.to(device)
        y = y.to(device).long()

        outputs = model(x)

        predicted = torch.argmax(outputs.data, 1)
        correct += (predicted == y).sum()
    accuracy = 100 * (correct.item()) / len(dataset)

    return accuracy


sparsity_level = None


def main(dataset, sparsity_level, bs):
    datax = torch.tensor(
        np.load(
            f"saved_files/{dataset}_train_coeff_{str(sparsity_level)}.npy",
            mmap_mode="r",
        )
    )
    datay = torch.tensor(
        np.load(
            f"saved_files/{dataset}_train_labels_{str(sparsity_level)}.npy",
            mmap_mode="r",
        )
    )

    train_ds = torch.utils.data.TensorDataset(datax, datay)

    datax_test = torch.tensor(np.load(f"saved_files/{dataset}_test_coeff_{sparsity_level}.npy", mmap_mode="r"))
    datay_test = torch.tensor(np.load(f"saved_files/{dataset}_test_labels_{sparsity_level}.npy", mmap_mode="r"))

    test_ds = torch.utils.data.TensorDataset(datax_test, datay_test)

    iter = 0

    while True:
        loss = train(train_ds, bs)
        acc = test(test_ds, bs)

        print("Epoch:", iter, "Train Loss:", loss, "Test accuracy:", acc)
        iter += 1

        if iter >= 1000:
            break
    torch.save(model, f"saved_files/{dataset}_model_{str(sparsity_level)}.pt")
    torch.save(optimizer, f"saved_files/{dataset}_optim_{str(sparsity_level)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sparsity_level", type=int)
    parser.add_argument("-bs", "--batch_size", type=int, default=1024)
    parser.add_argument(
        "-dataset",
        "--dataset_name",
        type=str,
        choices=["imagenet", "places365", "cub", "cifar10", "cifar100"],
    )
    args = parser.parse_args()

    model = get_model(args.dataset_name)
    model.to(device)

    # defining the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9)

    # defining Cross-Entropy loss
    criterion = torch.nn.CrossEntropyLoss()

    main(args.dataset_name, args.sparsity_level, args.batch_size)
