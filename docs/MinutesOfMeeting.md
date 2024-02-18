# Minutes of meeting - Clothes Extractor
This file should contain the information related to the different meetings that we are doing. The goal is not to have formal minutes of the meetings, but to have, at least, a summary of what has been done in each meeting so that, at a glance, people who have not been there understand what has been done.
## 2024-01-21
**Attendees:** Josep Maria y Raúl
Before staring with the different actions in the code I would like to proposse here a couple of things:
1. To create a folder for documentation (where this file is placed). We should use this folder to keep a more detailed information about the different parts of the application we are generating and eto explain some technical decissions. We can create there all the necessary markdown files (the names has to be clear and indicative about what they explain)
2. To create a folder that will contain all the colab/jupyter files that we are using. The own file has to contain an explanation of what it is doing and why has been created.

These are proposal that I think can help us to keep the project under control and write part of the documentation we will need to generate for delivering the project. **Comments and new propossals will be welcome.**

### During this meeting
* we have reviewed the dataset and dataloader code that Alvaro prepared and uploaded on the GitHub repossitory.
* Approved the PR that Alvaro requested
* Merge the "create_first_dataset" branch with main one
* Test the Dataloader and Dataset classes
* Add the agnostic_mask to the return values of Dataset
* the "use_dataset.ipynb" contains some code to test the Dataset.
* we have observed that mask in agnostic-mask folder is not totally appropriated. The "image-parse-v3" contains a more detailed mask that can be used but it has to be separated from the rest of segmentation classes that appears in the same file.

### Actions
* Josep Maria to generate the code to parse the detailed mask from "image-parse-v3"
* Raúl to prepare the doc folder and continue testing Dataset.

## 2024-01-25
**Attendees** Alvaro, Joan, Josep Maria, Pol, Raúl
Periodic meeting for reviewing the actual status of the project
###During the meeting
* Review of what we have done duriong the week
  * Extract new mask
  * Center mask
* Reviewing curren resuls we have discussed about
  * Scale of images. is this necessary or not
  * Occlusions of clothing due to how the model is wearing her outfit
  * We will need to define a kind of data augmentation, flip, rotation and color variations are proposed. Other possibilities are partial occlussions but this will be defined later.
* We have checked the algoritm to traini our net from scratch.
  * Take model clothe using a detailled mask (image 7 in Josep Maria jupyter) Concatenate with final Clothes mask (image 3) and we have to predict the final image (image 2). As we mentkioned last day we are going to add the final image mask to the loss calciulation of the net.
* Pol has asked about the net we are propossing fot doing the work. We have discussed about U-net and we have been checking different models and analysing the U-net created fort the lab.
* Pol has explain what is the goal and what we need to show in  the critical review of next wedneday
* With respect to the trainning of the network we have disscusse two possibilities we have, or train a net from scdratch of to use some U-net pretrainned with Imagenet and do the transfer learning and fine tunning.
* Pol proposse SNR and SSIM as metrics to evaluate how much good is out algorithm.

### Actions
* Alvarto and Joan to use the lab U-net as base to create a net that we can train from scratch.
* Josep Maria to use a pretrainned U-net and do the transfer learning
* Raul to prepare an index of the presentation and an intial propossal. When this is done help to the rest.


## 2024-02-15
**Attendees** Alvaro, Joan, Josep Maria, Pol, Raúl
* Split the training data into a training set and a validation set. *** Joan or Raul or Alvaro ***
* wabdb uploads too many things. Find a way to upload only the best model. *** Alvaro ***
* Upload images for each epoch in to wandb. *** Joan ***
* Check if the SSMI data is normalized. *** Raul ***
* Investigate ways to compare images to make Raul happier *** Raul ***
* Rerun the unet model with the unnormalized data. Wait for SSMI investigation to finish. *** Joan ***
* Write the report of the project in the README.md file.  *** ALL ***
* Take a look at the PatchGAN's discriminator. *** Josep ***
* Make an effort to show the results of the work/trains we are performing. *** ALL ***
* Include the model in the stored data to avoid conflicts when loading the weights. *** Alvaro ***