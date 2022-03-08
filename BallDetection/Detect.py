import torch
import numpy as np
import cv2
from PIL import Image
from Model import get_model_instance_segmentation

#Get model from class
num_classes = 3
model = get_model_instance_segmentation(num_classes)

#Load model
device = torch.device('cpu')
model.load_state_dict(torch.load("model", map_location=device))

#Move model to the right device
model.to(device)

#Switch to eval mode
model.eval()

def Detect(img):
    print(img.shape)
    with torch.no_grad():
        #rgb_img = Image.fromarray(img, 'RGB')
        rgbimg = cv2.cvtColor(img, cv2.COLOR_HSV2RGB).transpose(2, 0, 1)/255
        rgb_img = torch.as_tensor(rgbimg, dtype=torch.float32)
        prediction = model([rgb_img.to(device)])
        boxes = prediction[0]['boxes'][0].byte().cpu().numpy().copy()
        labels = prediction[0]['labels'][0].byte().cpu().numpy().copy()
        mask = prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy().copy()

        return boxes, labels, mask

if __name__ == '__main__':
    import ParseData
    img, _, _ = ParseData.get()
    _, _, mask = Detect(cv2.cvtColor(np.array(img[0]), cv2.COLOR_RGB2HSV))
    cv2.imshow("Yay", mask)
    cv2.waitKey(0)
