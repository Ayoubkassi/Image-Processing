#import tools

import sys
absolut_path = "/home/ayoub/Desktop/Image-Processing/"
sys.path.append(absolut_path+"Mask_RCNN/mrcnn")
from m_rcnn import *




#Load and prepare dataset

# Extract Images

annotations_path = absolut_path+"annotations.json"



#train dataset

dataset_train = load_image_dataset(os.path.join("/", annotations_path), absolut_path+"dataset", "train")
dataset_val = load_image_dataset(os.path.join("/", annotations_path), absolut_path+"dataset", "val")
class_number = dataset_train.count_classes()
print('Train: %d' % len(dataset_train.image_ids))
print('Validation: %d' % len(dataset_val.image_ids))
print("Classes: {}".format(class_number))


# Load image samples
#display_image_samples(dataset_train)



#training dataset_train
# get configuration
config = CustomConfig(class_number)
#testing configuration 
model = load_training_model(config)

#  Training Begin

train_head(model, dataset_train, dataset_train, config)

#test our model on images

# Load Test Model
# we trained 5 so we gonna load the number 5 
test_model, inference_config = load_test_model(class_number)

# Test on image
test_random_image(test_model, dataset_val, inference_config)



