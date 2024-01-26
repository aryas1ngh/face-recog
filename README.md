# face-recog
A simple face recognition model using OpenCV and Haar Cascades, using Local Binary Pattern Histogram (LBPH) algorithm.

### Clone the repo
- Execute in terminal :
```bash
git clone https://github.com/aryas1ngh/face-recog.git
```

### Files and directories
- ```cascades/```: contains the Haar cascades for face recogintion. Here, only ```haarcascade_frontalface_default.xml``` is included, but you can include more using this [Haar Cascade repository](https://github.com/opencv/opencv/tree/master/data/haarcascades).
- ```data/```: Used to store captured images which will further be used for training by the model.
- ```trainer/```: Used to store the ```lbph_trainer.yml``` trained model.
- ```collect.py```: Captures pictures which is used by model to train the classifier. Enter user ID at the time of execution. Default number of photos captured is 100, and can be changed.
- ```train.py```: Trains the model on the captured faces, labelled by IDs.
- ```main.py```: Live face recognition module with confidence percentage displayed.
- ```clean.bat```: Utility Windows batchfile used to empty the ```data/``` directory for training in newer settings.

### Workflow
- Collect data (pictures of users) by running ```collect.py``` on a webcam-enabled device.
- Train the facial image data with frontal-face Haar Cascade using ```train.py```.
- Run the ```main.py``` file for real-time face recognition with confidence percentage score.
  
