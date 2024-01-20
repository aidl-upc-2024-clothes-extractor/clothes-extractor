import sys
import argparse

from dataset_loader import DatasetUtil



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--download-dataset", help="Download the dataset and prepare it", action='store_true')

    args = parser.parse_args()
    
    if args.download_dataset:
        dataset_utils = DatasetUtil()
        dataset_utils.start()

