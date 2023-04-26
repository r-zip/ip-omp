import numpy as np
import torch
import clip
import pdb
from PIL import Image
import torchvision
from copy import copy
from collections import defaultdict
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LogisticRegression
import torch.multiprocessing
import CUB_dataset

torch.set_num_threads(1)
torch.multiprocessing.set_sharing_strategy('file_system')

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)



def get_data(transform, dataset):

    batch_size = 1

    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=1)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)

    elif dataset == "cub":


    return trainloader, testloader

def get_concepts(filename):
    list_of_concepts = []

    f = open(filename, "r")
    for line in f.readlines():
        list_of_concepts.append(line.strip())

    return list_of_concepts

def show_top_concepts(probs, concepts, topk = 5):
    indices = probs.argsort(descending=True)

    for index in indices[:topk]:
        print (concepts[index])

def print_concepts(concepts, indices, k=5):
    for index in indices[:k]:
        print(concepts[index])

def print_energy(concepts,indices, coeff):
    k = 0
    for index in indices:
        k += 1
        print(concepts[index], coeff[index].item())

def projection(Phi_t: torch.Tensor, perp: bool = False) -> torch.Tensor:
    U, *_ = torch.linalg.svd(Phi_t, full_matrices=False)
    P = U @ U.T

    if perp:
        return torch.eye(P.shape[0]).to(device) - P

    return P

def omp_estimate_y(Phi: torch.Tensor, y: torch.Tensor, indices):
    Phi_t = Phi[:, indices]
    return projection(Phi_t, perp=False) @ y


def ip_estimate_y(Phi: torch.Tensor, y: torch.Tensor, indices):
    return omp_estimate_y(Phi, y, indices)


def omp_estimate_x(Phi: torch.Tensor, y: torch.Tensor, indices) -> torch.Tensor:
    Phi_t = Phi[:, indices]
    x_hat = torch.zeros((Phi.shape[1])).to(device)
    x_hat[indices] = torch.linalg.pinv(Phi_t) @ y
    return x_hat

def ip_estimate_x(Phi: torch.Tensor, y: torch.Tensor, indices) -> torch.Tensor:
    return omp_estimate_x(Phi, y, indices)

def mutual_coherence(Phi: np.ndarray) -> float:
    plt.hist(np.abs(np.triu(Phi.T @ Phi, k=1)))
    plt.savefig("useless.png")
    return np.max(np.abs(np.triu(Phi.T @ Phi, k=1))).item()

def ip_objective(Phi: torch.Tensor, y: torch.Tensor, indices) -> torch.Tensor:
    P = projection(Phi[:, indices], perp=True)

    Phi_projected = P @ Phi
    Phi_projected_normalized = Phi_projected / torch.linalg.norm(
        Phi_projected, axis=0
    ).reshape(1, -1)

    objective = torch.absolute(Phi_projected_normalized.T @ y)
    objective[indices] = -np.inf
    return objective

def omp(Phi: np.ndarray, y: torch.Tensor, tol: float = 1e-6) -> dict:
    log = defaultdict(list)
    indices = []
    k = 0
    while True:
        P = projection(Phi[:, indices], perp=True)
        residual = P @ y

        # TODO: rethink termination criterion
        squared_error = residual.T @ residual
        if squared_error < tol or k == Phi.shape[1]-1:
            break

        objective = torch.absolute(Phi.T @ residual)
        log["objective"].append(objective.max().item())
        indices.append(torch.argmax(objective).item())
        log["indices"]= copy(indices)
        # y_hat = omp_estimate_y(Phi, y, indices)
        # log["y_hat"].append(y_hat)
        # x_hat = omp_estimate_x(Phi, y, indices)
        # log["x_hat"].append(x_hat)
        k += 1

    return dict(log)


def ip(Phi: torch.Tensor, y: torch.Tensor, tol: float = 1e-6, num_iterations = None):
    log = defaultdict(list)
    indices = []
    k = 0
    while True:
        objective = ip_objective(Phi, y, indices=indices)
        max_objective = objective.max()
        log["objective"].append(max_objective.item())

        # TODO: rethink termination criterion
        if num_iterations is None and (torch.absolute(max_objective) < tol or k == Phi.shape[1]-1):
            break
        elif k == num_iterations:
            break

        indices.append(torch.argmax(objective).item())
        log["indices"] = copy(indices)
        # y_hat = ip_estimate_y(Phi, y, indices)
        # log["y_hat"].append(y_hat)
        # x_hat = ip_estimate_x(Phi, y, indices)
        # log["x_hat"].append(x_hat)
        k += 1

    return dict(log)

def mse(estimated: torch.Tensor, true: torch.Tensor):
    return torch.mean((estimated - true) ** 2).item()

def analysis_by_synthesis(dataloader, dictionary):

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    with torch.no_grad():
        iteration = 0

        datax, datay = [], []
        for data in dataloader:
            if iteration%100 == 0:
                print (iteration)

            # get the inputs; data is a list of [inputs, labels]
            image, labels = data
            image = image.to(device)

            image_features = model.encode_image(image).float()
            image_features = image_features / torch.linalg.norm(image_features, axis=1).reshape(-1, 1)

            # print ()
            log = ip(Phi = dictionary, y = image_features[0], tol=1e-2, num_iterations=35)
            # print_concepts(concepts, log["indices"], k=10)
            # y_hat = ip_estimate_y(dictionary, image_features[0], log["indices"])
            # print (len(log["indices"]), mse(y_hat, image_features[0]))

            coeffs = ip_estimate_x(dictionary, image_features[0], log["indices"])
            # print_energy(concepts, log["indices"], x)

            datax.append(coeffs.clone())
            datay.append(labels)
            iteration += 1

            # if iteration == 500:
            #     break

    datax = torch.stack(datax).cpu().numpy()
    datay = torch.stack(datay).cpu().numpy().squeeze()
    return datax, datay

def main():
    dataset = "cub" #cifar10
    trainloader, testloader = get_data(preprocess, dataset)

    concepts = get_concepts("concept_sets/" + dataset + ".txt")
    text = clip.tokenize(concepts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        dictionary = text_features.T.float()

        dictionary = dictionary / torch.linalg.norm(dictionary, axis=0)
        print(mutual_coherence(dictionary.cpu().numpy()))

    datax, datay = analysis_by_synthesis(trainloader, dictionary)

    np.save(dataset + '_train_coeff.npy', datax)
    np.save(dataset + '_train_labels.npy', datay)
    #train logistic regression classifier
    clf = LogisticRegression(random_state=0, multi_class='multinomial').fit(datax, datay)

    datax_test, datay_test = analysis_by_synthesis(testloader, dictionary)
    np.save(dataset + '_test_coeff.npy', datax_test)
    np.save(dataset + '_test_labels.npy', datay_test)

    print (clf.score(datax_test, datay_test))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
