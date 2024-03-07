# clothes-extractor
2024 Postgraduate course on Artificial Intelligence with Deep Learning, **UPC School**.

Authored by **Josep Maria Salvia Hornos, Álvaro Cabanas Martí, Joan Puig Sanz and Raúl Alares**.

Advised by **Pol Caselles**.

---

Table of Contents
=================
  * [INTRODUCTION](#introduction)
  * [DATASET](#dataset)
  * [PROJECT ARCHITECTURE](#project-architecture)
	 * [INITIAL APPROACH](#initial-approach)

## Motivation
We want to allow users to try virtually the clothes before buying them online.
This could help users to make better decisions and save returning costs.

![docs/images/final_objective.png](docs/images/final_objective.png)

## Our proposal
We did not have enough resources to implement the whole pipeline during given time, so we modified the initial proposal to a more feasible one.

In this project we have trained a model using a patch-GAN to extract the t-shirt from an image of a person and generate a frontal img of the t-shirt.

On inference the model expects an input image and a segmented mask from the image. We have left the segmenter model that would extract the mask for future development.

![docs/images/first_pipeline.png](docs/images/first_pipeline.png)


## Future pipeline
For this workflow we are defining 2 pipelines:

First pipeline:

A Segmenter generates and img mask for the different clothes in the source img.
An img2img model based on a patch-GAN (U-Net for the generator) uses the image mask and the source to generate a frontal img of the extracted t-shirt.

Second pipeline:
Use the VITTON HD model to infer our generated T-shirt picture and the destiny.

![docs/images/two_pipelines.png](docs/images/two_pipelines.png)


## Dataset
We use the dataset from the [VITON-HD project](https://github.com/shadow2496/VITON-HD).
In order to run the code you need to download and unzip the [VITON-HD dataset](https://github.com/shadow2496/VITON-HD?tab=readme-ov-file#dataset) in the `data` folder.


## Project architecture
### Initial approach
We wanted to iterate and not start  by steps so we

[UNET](https://arxiv.org/abs/1505.04597)
[segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

### Final architecture: PATCH-GAN (UNET Generator + Discriminator)
[]https://arxiv.org/pdf/1611.07004v3.pdf


## Development
### Low res dataset
For development purposes there is the possibility to downsize the dataset to a given size and reduce the number of images.

The viton dataset should be downloaded and unzipped in the `data` folder.

Running the following command, we will create a folder `zalando-low-res` with our downsized images.
The flag `-f` will set the number of files (or empty for all) and the flag `-s` the size: 

```bash
$ python dataset/down_sample_dataset.py -f 20 -s 64
```
