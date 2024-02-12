import os
from pathlib import Path
import argparse
import pingouin as pg

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file1', type=str, default='./ssim_3.txt')
    parser.add_argument('--file2', type=str, default='./ssim_5.txt')

    opt = parser.parse_args()
    return opt

def read_data_file(name):
    list1 = []
    f = open(name, "rt")
    content=f.readline()
    while content:
        list1.append(float(content))
        content=f.readline().strip()
    f.close()
    return list1


def main(opt):
    list1 = read_data_file(opt.file1)
    list2 = read_data_file(opt.file2)

    print(pg.ttest(x=list1, y=list2, alternative='two-sided', correction=False))

if __name__ == '__main__':
    opt = get_opt()
    main(opt)