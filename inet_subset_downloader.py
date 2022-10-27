from clip_retrieval.clip_back import (
    load_clip_index,
    load_metadata_provider,
    ClipOptions,
    KnnService,
)

import numpy as np
import pandas as pd

import argparse
import base64
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download subset of ImageNet validation set"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of image neighborhoods to download",
    )
    parser.add_argument(
        "--num-neighbors", type=int, default=100, help="Number of neighbors to download"
    )
    parser.add_argument(
        "-indice-folder",
        type=str,
        default="./data",
        help="Folder containing the indice files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/datasets/ILSVRC2012/val-tars/",
        help="Output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    return args


def search_and_download(knn_service, filepath, args):
    # Search the index for the nearest neighbors of each image
    with open(filepath, "rb") as f:
        image = base64.b64encode(f.read()).decode("utf-8")

    url_captions = []
    multiplier = 2
    while len(url_captions < args.num_neighbors):
        results = knn_service.query(
            image_input=image,
            modality="image",
            indice_name="laion_400m",
            num_images=args.num_neighbors,
            num_result_ids=args.num_neighbors * multiplier,
        )
        url_captions = pd.DataFrame(
            [
                (e["url"], e["caption"])
                for e in filter(lambda x: x.get("url") and x.get("caption"), results)
            ],
            columns=["url", "caption"],
        )
        multiplier *= 2

    # Download the neighborhoods using img2dataset
    # Join the output directory and the image name to get the output path
    out_dir = os.path.join(args.output_dir, os.path.basename(filepath))[:-4]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print("Skipping", filepath)

    url_captions.to_parquet(os.path.join(out_dir, "metadata.parquet"))

    # Download the images
    os.system(
        f"img2dataset --input_format=parquet --url_list {os.path.join(out_dir, 'metadata.parquet')} --output_folder {out_dir} --processes_count=16 --thread_count=64 --output_format=webdataset --url_col=url --caption_col=caption"
    )


def main():
    # Parse arguments
    args = parse_args()

    # Turn ImageNet validation folder into a list of all the image paths
    val_dir = "~/datasets/ILSVRC2012/val/"
    val_dir = os.path.expanduser(val_dir)

    val_files = []
    for dir in os.listdir(val_dir):
        for file in os.listdir(os.path.join(val_dir, dir)):
            val_files.append(os.path.join(val_dir, dir, file))

    # Take a random subset of the images and set seed for reproducibility
    np.random.seed(args.seed)
    val_files = np.random.choice(val_files, args.num_images, replace=False)

    # Load clip index and metadata provider
    columns = ["url", "caption"]
    # metadata, _ = load_metadata_provider('./', True, False, 'image.index', columns, False)
    options = ClipOptions(
        indice_folder="./data",
        clip_model="ViT-B/32",
        enable_hdf5=True,
        enable_faiss_memory_mapping=True,
        columns_to_return=columns,
        reorder_metadata_by_ivf_index=False,
        enable_mclip_option=False,
        use_jit=False,
        use_arrow=False,
        provide_safety_model=False,
        provide_violence_detector=False,
        provide_aesthetic_embeddings=False,
    )
    resource = load_clip_index(options)
    knn_service = KnnService(clip_resources={"laion_400m": resource})

    # Loop through the images and download the neighborhoods and save progress to a json file
    progress = {"all_files": val_files, "current_file": 0}
    for i, filepath in enumerate(val_files):
        print("Downloading neighborhood", i, "of", args.num_images)
        search_and_download(knn_service, filepath, args)
        progress["current_file"] = i
        with open("progress.json", "w") as f:
            json.dump(progress, f)


if __name__ == "__main__":
    main()
