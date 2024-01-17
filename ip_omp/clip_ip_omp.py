import os

# these lines are needed to prevent pytorch from using all CPU cores
os.system("export MKL_NUM_THREADS=4")
os.system("export NUMEXPR_NUM_THREADS=4")
os.system("export OMP_NUM_THREADS=4")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import clip  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import tqdm  # noqa: E402

from . import algorithms  # noqa: E402
from . import util  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
torch.set_num_threads(1)


def get_sparse_code(dataset, dataset_name, dictionary, num_iterations, bs):
    batch_size = bs

    with torch.no_grad():
        iteration = 0
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        datax = torch.empty((len(dataset), dictionary.shape[1]))
        datay = torch.empty((len(dataset)))

        element_index = 0
        for data in tqdm.tqdm(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            if dataset_name == "cub":
                image, labels, _ = data
            else:
                image, labels = data
            image = image.to(device)

            dictionary_batch = dictionary.repeat(image.size(0), 1, 1)

            if dataset_name in ["cub", "cifar10", "cifar100"]:
                image_features = model.encode_image(image).float()
                image_features = image_features / torch.linalg.norm(image_features, axis=1).reshape(-1, 1)
            else:
                image_features = image.clone()

            log = algorithms.ip(
                Phi=dictionary_batch,
                y=image_features.unsqueeze(2),
                num_iterations=num_iterations,
                device=device,
                compute_xhat=False,
            )

            coeffs = algorithms.estimate_x_lsq(
                dictionary_batch,
                image_features.unsqueeze(2),
                log["columns"],
                device=device,
            )  # estimate_x(dictionary_batch, image_features, log["indices"])

            datax[element_index : element_index + coeffs.size(0)] = coeffs.cpu().squeeze().clone()
            datay[element_index : element_index + coeffs.size(0)] = labels.cpu().clone()

            element_index = element_index + coeffs.size(0)

            iteration += 1

    return datax.numpy(), datay.numpy()


def main(dataset_name, sparsity_level, bs):
    dataset = dataset_name  # "cifar100" #"imagenet" #"places365" #"cub" #cifar10
    train_ds, test_ds = util.get_data(preprocess, dataset)
    module_dir = Path(__file__).parent

    concepts = util.get_concepts(module_dir / ("concept_sets/" + dataset + ".txt"))
    text = clip.tokenize(concepts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        dictionary = text_features.T.float()

        dictionary = dictionary / torch.linalg.norm(dictionary, axis=0)

        datax, datay = get_sparse_code(train_ds, dataset, dictionary, num_iterations=sparsity_level, bs=bs)

        np.save(module_dir / f"saved_files/{dataset}_train_coeff_{str(sparsity_level)}.npy", datax)
        np.save(module_dir / f"saved_files/{dataset}_train_labels_{str(sparsity_level)}.npy", datay)

        datax_test, datay_test = get_sparse_code(test_ds, dataset, dictionary, num_iterations=sparsity_level, bs=bs)

        np.save(module_dir / f"saved_files/{dataset}_test_coeff_{str(sparsity_level)}.npy", datax_test)
        np.save(module_dir / f"saved_files/{dataset}_test_labels_{str(sparsity_level)}.npy", datay_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sparsity_level", type=int)
    parser.add_argument("-bs", "--batch_size", type=int, default=128)
    parser.add_argument(
        "-dataset",
        "--dataset_name",
        type=str,
        choices=["imagenet", "places365", "cub", "cifar10", "cifar100"],
    )
    args = parser.parse_args()
    main(args.dataset_name, args.sparsity_level, args.batch_size)
