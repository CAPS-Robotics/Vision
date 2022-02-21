import json
import numpy as np
from PIL import Image

def get():
    images = []
    bounding_box_coords = []
    bounding_box_areas = []
    ball_types = []
    masks = []

    PATH = [
        "BallDetection/ball_data/ballr/bluebig_1.mp4",
        "BallDetection/ball_data/ballr/bluebig_2.mp4",
        "BallDetection/ball_data/ballr/f6nc2t.mp4",
        "BallDetection/ball_data/ballr/redbig_1.mp4",
        "BallDetection/ball_data/ballr/redbig_2.mp4",
        "BallDetection/ball_data/ballr/redtraining.mp4"
    ]

    for path in PATH:
        #Max 5075
        labels = []
        images_paths = []
        for i in range(5075 // 25):
            try:
                labels.append(json.load(open(path+"/ann/frame_" + "".join([str(0) for _ in range(5-len(str(25*i)))]) + str(25*i) + ".png.json")))
                images_paths.append(path+"/img/frame_" + "".join([str(0) for _ in range(5-len(str(25*i)))]) + str(25*i) + ".png")

            except FileNotFoundError: #No more files to load in that folder
                break

        for i in range(len(images_paths)):
            try:
                img = Image.open(images_paths[i]).convert("RGB")
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
    
    return images, masks


