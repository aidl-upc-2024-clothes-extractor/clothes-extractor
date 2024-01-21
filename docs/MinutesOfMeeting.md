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