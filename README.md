# Road-segmentation | Residual U-Net for aerial sattelite images road segmentation

### About the Project
In this project, we implemented and trained using satellite images, a residual
Unet for classifying pixels on an aerial image as either a road or
background.


### Contributors
- Dorian Popvic[@DorianPopovic](https://github.com/DorianPopovic)
- Lucien Barret [@lubacien](https://github.com/lubacien)
- Théophile Bouiller[@tbouiller](https://github.com/tbouiller)
- AiCrowd team name : theophileetlucienetdorian 
- AiCrowd best submission id : 31607 theophile_bouiller (890)
### Setup environment
The necessary can be installed using `pip` and the provided `requirements.txt`
```bash
   pip install -r requirements.txt
```
In order to run tf_aerial_images tensorflow must be downgraded. In order to get the correct version for tensorflow execute 
```bash
   pip install --upgrade tensorflow==1.13
```
## Code architecture
In this section we explain how our project folder is organised and where to find the needed files.

### Data Folder
This folder contains the train and test data in zip archives. In order to run the code the dataset consisting of training.zip and test_set_images.zip must be placed in the data/ folder, due to limits to the submission size. Executing run.py will unzip these files and make the the following directories containing the files:

1. **training** : consists of the images and associated groundtruths.
2. **test_set_images** : Consists of the images we use to test our model.

### Scripts folder
All our project implementations can be found inside the /scripts folder.$

**Python executables .py**

1. **run.py**: Generates the predicted ground truth using the weights that gave us our best prediction on Aicrowd.
2. **train_unet.py**: Trains the unet and creates new weights.
3. **preprocessing.py**: Functions that help us preprocess the data.
4. **image_processing.py**: Applies modifiers to the the output images.

### Run
To generate our best prediction submitted on aicrowd, execute:
```bash
   python3 run.py
```
from the root directory of the project. This will generate the predictions using the given weights.
The predictions are saved in the folder /submissionimages, a csv submission file will be created under the name submission.csv in the data/ directory.

To train the model and build the weights, execute:

```bash
   python3 train_unet.py
```
### Process the output images
In order to execute a floodfill on the prediction execute:
```bash
   python3 image_processing.py
```
The floodfilled images will be saved in a new folder named sub_mod/. a new folder named mont_dir/ will contain the grouped images from the test set, predictions and floodfilled images.
