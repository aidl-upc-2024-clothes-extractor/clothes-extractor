import argparse
from dataset import ClothesDataset, ClothesDataLoader
from config import Config
from argparse_dataclass import ArgumentParser

def main():
    args = ArgumentParser(Config)
    cfg = args.parse_args()
    print(cfg.batch_size)

    dataset = ClothesDataset(cfg)
    loader = ClothesDataLoader(dataset, cfg.batch_size, num_workers=cfg.workers)

    # FIXIT: This is just to test the loader works
    for batch_idx, result in enumerate(loader.next_batch()):
        print(batch_idx)
        print(result)


if __name__ == '__main__':
    main()
