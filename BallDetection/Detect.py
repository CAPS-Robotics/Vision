import torch
from PIL import Image
from Model import get_model_instance_segmentation

#Get model from class
num_classes = 3
model = get_model_instance_segmentation(num_classes)

#Load model
model.load_state_dict(torch.load("model"))

#Move model to the right device
device = torch.device('cpu')
model.to(device)

#Switch to eval mode
model.eval()

def Detect():
    with torch.no_grad():
        prediction = model([img.to(device)])
        img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        boxes = prediction[0]['boxes'][0].byte().cpu().numpy()
        labels = prediction[0]['labels'][0].byte().cpu().numpy()
        mask = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())

        return boxes, labels, mask