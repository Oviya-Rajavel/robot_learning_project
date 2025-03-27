# Robot Learning Project
The task board urdf is based on this [Github page](https://github.com/hrii-iit/robothon-2023-board-description)

### Installing Dependencies / Replicating the environment:
* To create the same environment, the following command can be used:
`conda create -n <environment-name> --file requirements.txt`
or alternatele, the following command `pip install -r requirements.txt`

* Copying the robothon taskboard file to the location of the rlbench library:
Two additional files need to be copied to the location where rlbench is installed in your anaconda/miniconda environment. They are located in rlbench task files folder.
`taskboard_robothon.py` to the `task` folder; and `taskboard_robothon.ttm` to the `task_ttms` folder

### Dataset
The dataset can be downloaded from the following location: [Dataset Link](https://drive.google.com/drive/folders/1t4bJptSzxMFEsYImzCt1GaGbGiiZ8l7o?usp=sharing) 

### Training the CNN model:
The file to be run is `imitation_learning.py`. Dataset location should be specified in `imitation_learning.py` in the environment creation, with the parameter: dataset_root

### Training the CNN + Transformer model:
The file to be run is `imitation_learning_transformer.py`. Dataset location should be specified in `imitation_learning_transformer.py` in the environment creation, with the parameter: dataset_root

### Fine tuning the already trained (CNN+ Transformer) model:
The file `dagger.py` has to be run to fine-tune the trained model
