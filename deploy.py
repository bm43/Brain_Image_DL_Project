import torch
import monai

PATH = "C:/Users/SamSung/Desktop/pers_research/kaggle/mri-resnet10/weights/3d-resnet10_T1wCE_fold1_0.573.pth"

model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=1)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()
