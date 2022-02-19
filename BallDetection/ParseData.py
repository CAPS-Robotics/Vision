import json
import numpy as np
import cv2
from PIL import Image

labels = [json.load(open("BallDetection/Data/ann/frame_" + "".join([str(0) for _ in range(5-len(str(25*i)))]) + str(25*i) + ".png.json")) for i in range(4825 // 25)]
images_paths = ["BallDetection/Data/img/frame_" + "".join([str(0) for _ in range(5-len(str(25*i)))]) + str(25*i) + ".png" for i in range(4825 // 25)]

images = []
bounding_box_coords = []
bounding_box_areas = []
ball_types = []
masks = []
for i in range(len(images_paths)):
    try:
        img = Image.open(images_paths[i]).convert("HSV")
        img = img.resize((1080, 720))

        (x0, y0), (x1, y1) = labels[i]["objects"][0]["points"]["exterior"]

        ball_type = labels[i]["objects"][0]["classTitle"]
        if ball_type == "blueball":
            ball_type = 1
        elif ball_type == "redball":
            ball_type = 2
        else:
            raise ValueError #Unrecognized ball type

        #Rescale coords to 1080 by 720
        x0 = int(x0 * 1080/1280)
        x1 = int(x1 * 1080/1280)

        #Create image mask using bounding box
        mask = np.zeros((720, 1080))
        mask[y0:y1, x0:x1] = 1
        mask *= ball_type #Make mask numbers match ball_type 

        images.append(img)
        bounding_box_coords.append([x0, y0, x1, y1])
        bounding_box_areas.append(abs((x1-x0)*(y1-y0)))
        ball_types.append(ball_type)
        masks.append(mask)

        #cv2.imshow("Image", img)
        #cv2.waitKey(0)

    except IndexError:
        print("Incomplete label")

if __name__ == '__main__':
    for i in range(len(masks)):
        mask = masks[i]

        obj_ids = np.unique(mask)
        #obj_ids = obj_ids[1:] #Remove the background from the ids

        imasks = (mask == obj_ids[:, None, None]) * 1

        img = Image.fromarray(imasks)
        img.show()

