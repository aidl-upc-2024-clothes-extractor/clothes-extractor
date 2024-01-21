import argparse
from dataset import ClothesDataset, ClothesDataLoader


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--dataset_dir', type=str, default='./datasets/')
    parser.add_argument('--dataset_list', type=str, default='test_pairs.txt')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('--load_height', type=int, default=1024)
    parser.add_argument('--load_width', type=int, default=768)

    opt = parser.parse_args()
    return opt


def main():
    opt = get_opt()
    print(opt)

    dataset = ClothesDataset(opt)
    loader = ClothesDataLoader(opt, dataset)

    # FIXIT: This is just to test the loader works
    for batch_idx, result in enumerate(loader.next_batch()):
        print(batch_idx)
        print(result)


if __name__ == '__main__':
    main()
