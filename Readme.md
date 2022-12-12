# Open Source Software Term Project
###### 202235028 김진태, 202235071 양민혁, 202235107 이호찬, 202235153 황민우

```sh 
# install pip model
! pip install git+https://github.com/divamgupta/image-segmentation-keras 
 # download the dataset
! wget https://github.com/divamgupta/datasets/releases/download/seg/dataset1.zip && unzip dataset1.zip 
#Initialize the model
from keras_segmentation.models.unet import vgg_unet
model = vgg_unet(n_classes=50 ,  input_height=320, input_width=640  )

#Train the model
model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5  
)
out = model.predict_segmentation(
    inp="dataset1/images_prepped_test/0016E5_07965.png",
    out_fname="/tmp/out.png"
)
%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt
plt.imshow(out)
from IPython.display import Image
Image('/tmp/out.png')
o = model.predict_segmentation(
    inp="dataset1/images_prepped_test/0016E5_07965.png",
    out_fname="/tmp/out.png" , overlay_img=True, show_legends=True,
    class_names = [ "Sky",    "Building", "Pole","Road","Pavement","Tree","SignSymbol", "Fence", "Car","Pedestrian", "Bicyclist"]
)
from IPython.display import Image
Image('/tmp/out.png')
```