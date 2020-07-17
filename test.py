from __future__ import print_function, division

from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, Normalize, RandomRotation, RandomAffine,RandomHorizontalFlip,RandomVerticalFlip,ToTensor
from models.cbamdensenet import cbam_densenet169
import numpy as np
plt.ion()  # interactive mode
# Model storage path
model_save_path = '/C-DenseNet/results/C-DenseNet/best_model.pth'  # type:

# Define pretraining transformation
preprocess_transform = transforms.Compose([
    Resize(size=(640, 640)),
    transforms.ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
class_names = ['0', '1', '2', '3', '4', '5']  # This order is very important. It should be consistent with the class name order during training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ------------------------ Load the model --------------------------- #
model = cbam_densenet169()
model.eval()
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(model_save_path))
model = model.cuda()
model.eval()



image_PIL = Image.open('/C-DenseNet/dataset/test/one.jpg')
image_tensor = preprocess_transform(image_PIL)
image_tensor.unsqueeze_(0)
# There is no such sentence will report an error
image_tensor = image_tensor.to(device)

out = model(image_tensor)
# The prediction results are obtained and sorted from large to small
_, indices = torch.sort(out, descending=True)
# Returns the percentage of each predicted value
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100


result1 = np.arange(6)
for i in range(6):
    result1[i] = indices[0][i]


print(result1[0])