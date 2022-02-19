import os
import numpy as np
import torch
import transforms as T
import ParseData

class BallDataset(torch.utils.data.Dataset):
    def __init__(self, transforms):
        self.transforms = transforms
        self.imgs = ParseData.images
        self.masks = ParseData.masks

    def __getitem__(self, idx):
        img = self.imgs[idx]
        mask = self.masks[idx]

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:] #Remove the background from the ids

        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])


        #Convert to torch.Tensor
        #boxes = torch.as_tensor(ParseData.bounding_box_coords, dtype=torch.float32)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #labels = torch.as_tensor(ParseData.ball_types, dtype=torch.int64)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        #Idk what this is
        iscrowd = torch.zeros((2,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

