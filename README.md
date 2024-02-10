# clothes-extractor
Pytorch deep learning project to generate the frontal view of the upper clothes a person is wearing on an inferred image. 

Using a Segmenter and Image2Image models.

# Dataset
We use the dataset from the [VITON-HD project](https://github.com/shadow2496/VITON-HD).
In order to run the code you need to download and unzip the [VITON-HD dataset](https://github.com/shadow2496/VITON-HD?tab=readme-ov-file#dataset) in the `data` folder.

# Low res dataset
For development purposes there is the possibility to downsize the dataset to a given size and reduce the number of images.

The viton dataset should be downloaded and unzipped in the `data` folder.

Running the following command, we will create a folder `zalando-low-res` with our downsized images.
The flag `-f` will set the number of files (or empty for all) and the flag `-s` the size: 

```bash
$ python dataset/down_sample_dataset.py -f 20 -s 64
```
