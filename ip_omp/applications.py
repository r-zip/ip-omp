import clip
import numpy as np
import torch
import torch.multiprocessing
import torchvision
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from .algorithms import ip, ip_estimate_x, mutual_coherence
from .constants import DEVICE

torch.set_num_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")


model, preprocess = clip.load("ViT-B/16", device=DEVICE)


def get_data(transform, dataset):
    batch_size = 1

    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=1
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=1
        )
    else:
        raise ValueError("dataset not understood.")

    return trainloader, testloader


def get_concepts(filename):
    list_of_concepts = []

    f = open(filename, "r")
    for line in f.readlines():
        list_of_concepts.append(line.strip())

    return list_of_concepts


def show_top_concepts(probs, concepts, topk=5):
    indices = probs.argsort(descending=True)

    for index in indices[:topk]:
        print(concepts[index])


def print_concepts(concepts, indices, k=5):
    for index in indices[:k]:
        print(concepts[index])


def print_energy(concepts, indices, coeff):
    k = 0
    for index in indices:
        k += 1
        print(concepts[index], coeff[index].item())


def analysis_by_synthesis(dataloader, dictionary):
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    with torch.no_grad():
        iteration = 0

        datax, datay = [], []
        for data in dataloader:
            if iteration % 100 == 0:
                print(iteration)

            # get the inputs; data is a list of [inputs, labels]
            image, labels = data
            image = image.to(DEVICE)

            image_features = model.encode_image(image).float()
            image_features = image_features / torch.linalg.norm(
                image_features, axis=1
            ).reshape(-1, 1)

            # print ()
            log = ip(Phi=dictionary, y=image_features[0], tol=1e-2, num_iterations=35)
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
    dataset = "cifar10"
    trainloader, testloader = get_data(preprocess, dataset)

    concepts = get_concepts("concept_sets/" + dataset + ".txt")
    text = clip.tokenize(concepts).to(DEVICE)

    with torch.no_grad():
        text_features = model.encode_text(text)
        dictionary = text_features.T.float()

        dictionary = dictionary / torch.linalg.norm(dictionary, axis=0)
        print(mutual_coherence(dictionary.cpu().numpy()))

    datax, datay = analysis_by_synthesis(trainloader, dictionary)

    np.save(dataset + "_train_coeff.npy", datax)
    np.save(dataset + "_train_labels.npy", datay)
    # train logistic regression classifier
    clf = LogisticRegression(random_state=0, multi_class="multinomial").fit(
        datax, datay
    )

    datax_test, datay_test = analysis_by_synthesis(testloader, dictionary)
    np.save(dataset + "_test_coeff.npy", datax_test)
    np.save(dataset + "_test_labels.npy", datay_test)

    print(clf.score(datax_test, datay_test))


if __name__ == "__main__":
    main()
