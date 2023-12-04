

import os

#these lines are needed to prevent pytorch from using all CPU cores
os.system("export MKL_NUM_THREADS=4")
os.system("export NUMEXPR_NUM_THREADS=4")
os.system("export OMP_NUM_THREADS=4")

import numpy as np
import torch
import tqdm
from copy import copy
import algorithms
import pdb
import util
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
print (device)
torch.set_num_threads(1)


def get_sparse_code(dataset, dataset_name, dictionary, num_iterations):

    batch_size = 128

    with torch.no_grad():
        iteration = 0
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        datax = torch.empty((len(dataset), dictionary.shape[1]))
        datay = torch.empty((len(dataset)))

        element_index = 0
        for data in tqdm.tqdm(dataloader):

            # get the inputs; data is a list of [inputs, labels]
            if  dataset_name == "cub":
                image, labels, _ = data
            else:
                image, labels = data
            image = image.to(device)


            dictionary_batch = dictionary.repeat(image.size(0), 1, 1)

            if  dataset_name in ["cub", "cifar10", "cifar100"]:
                image_features = model.encode_image(image).float()
                image_features = image_features / torch.linalg.norm(image_features, axis=1).reshape(-1, 1)
            else:
                image_features = image.clone()

            log = algorithms.ip(Phi = dictionary_batch, y = image_features.unsqueeze(2), num_iterations=num_iterations, device=device, compute_xhat=False)


            coeffs = algorithms.estimate_x_lsq(dictionary_batch, image_features.unsqueeze(2), log["columns"], device=device) #estimate_x(dictionary_batch, image_features, log["indices"])


            datax[element_index:element_index+coeffs.size(0)] = coeffs.cpu().squeeze().clone()
            datay[element_index:element_index + coeffs.size(0)] = labels.cpu().clone()

            element_index = element_index + coeffs.size(0)

            iteration += 1


    return datax.numpy(), datay.numpy()



def main():
    dataset = "cifar10" #"imagenet" #"places365" #"cub" #cifar10
    train_ds, test_ds = util.get_data(preprocess, dataset)

    concepts = util.get_concepts("concept_sets/" + dataset + ".txt")
    text = clip.tokenize(concepts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        dictionary = text_features.T.float()

        dictionary = dictionary / torch.linalg.norm(dictionary, axis=0)

        datax, datay = get_sparse_code(train_ds, dataset, dictionary, num_iterations=10)
        
    pdb.set_trace()


if __name__ == '__main__':
    main()
