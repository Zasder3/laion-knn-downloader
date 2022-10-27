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
        default=2,
        help="Number of image neighborhoods to download",
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=32768,
        help="Number of neighbors to download",
    )
    parser.add_argument(
        "-indice-folder",
        type=str,
        default="./data",
        help="Folder containing the indice files",
    )
    parser.add_argument(
        "--imagenet-dir",
        type=str,
        default="~/datasets/ILSVRC2012/val/",
        help="Folder containing the ImageNet validation set",
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
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a file containing a list of already downloaded images",
    )
    args = parser.parse_args()

    # Expand user paths
    args.indice_folder = os.path.expanduser(args.indice_folder)
    args.imagenet_dir = os.path.expanduser(args.imagenet_dir)
    args.output_dir = os.path.expanduser(args.output_dir)
    args.resume_from = os.path.expanduser(args.resume_from)
    return args


def search_and_download(knn_service, filepath, args):
    # Search the index for the nearest neighbors of each image
    with open(filepath, "rb") as f:
        image = base64.b64encode(f.read()).decode("utf-8")

    url_captions = []
    multiplier = 2
    while len(url_captions) < args.num_neighbors:
        if multiplier > 2:
            print("Doubling query size")
            print("Only found", len(url_captions), "neighbors")
        elif multiplier > 4:
            break
        results = knn_service.query(
            image_input=image,
            modality="image",
            indice_name="laion_400m",
            num_images=args.num_neighbors * multiplier,
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
    url_captions = url_captions.iloc[: min(args.num_neighbors, len(url_captions))]

    # Download the neighborhoods using img2dataset
    # Join the output directory and the image name to get the output path
    *_, class_name, image_name = filepath.split("/")
    out_dir = os.path.join(args.output_dir, class_name, image_name)
    out_dir = out_dir[: out_dir.rfind(".")]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print("Skipping", filepath)

    url_captions.to_parquet(os.path.join(out_dir, "metadata.parquet"))

    # Download the images
    os.system(
        f"img2dataset --input_format=parquet --url_list {os.path.join(out_dir, 'metadata.parquet')} --output_folder {out_dir} --processes_count=16 --thread_count=64 --output_format=webdataset --url_col=url --caption_col=caption --number_sample_per_shard 1000"
    )


def main():
    # Parse arguments
    args = parse_args()

    # Turn ImageNet validation folder into a list of all the image paths
    print("Loading ImageNet validation set")

    val_files = []
    for dir in os.listdir(args.imagenet_dir):
        for file in os.listdir(os.path.join(args.imagenet_dir, dir)):
            val_files.append(os.path.join(args.imagenet_dir, dir, file))

    # Take a random subset of the images and set seed for reproducibility
    np.random.seed(args.seed)
    val_files = list(np.random.choice(val_files, args.num_images, replace=False))

    # Load clip index and metadata provider
    print("Loading index")
    columns = ["url", "caption"]
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
    if args.resume_from:
        print("Resuming from", args.resume_from)
        with open(args.resume_from, "r") as f:
            progress = json.load(f)
        val_files = progress["all_files"][progress["current_file"] :]
    else:
        progress = {"all_files": val_files, "current_file": 0}

    for i, filepath in enumerate(val_files, start=progress["current_file"]):
        print("Downloading neighborhood", i + 1, "of", args.num_images)
        search_and_download(knn_service, filepath, args)
        progress["current_file"] = i + 1
        with open(os.path.join(args.output_dir, "progress.json"), "w") as f:
            json.dump(progress, f)


if __name__ == "__main__":
    main()
