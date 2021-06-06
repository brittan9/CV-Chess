import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from bounding_box import BoundingBox
from detect import detect

# globals for ML curve fitting
angle = nn.Parameter(torch.randint(75, (1,), dtype=torch.float64), requires_grad=True)
velocity = nn.Parameter(torch.randint(50, (1,), dtype=torch.float64), requires_grad=True)

def pixelLocationsToCoordinates(moving_pixel_locations, num_data_pts, height):
    points = np.empty(shape=[num_data_pts, 2])
    curr_idx = 0
    for frame in moving_pixel_locations:
        for pixel_location in frame:
            points[curr_idx, 0] = pixel_location[1]
            points[curr_idx, 1] = height - pixel_location[0]
            curr_idx += 1

    return points

def plotMovingPixels(coordinates):
    plt.scatter(coordinates[:, 0], coordinates[:, 1])
    plt.savefig("shot_graph")

def flipBoundingBox(bbox, frame_width):
    return BoundingBox(frame_width - bbox.x2, bbox.y1, frame_width - bbox.x1, bbox.y2)

def locatePerson(img_path):
    # person detect
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.conf = 0.75
    model.classes = [0]
    results = model(img_path, size=544)
    return BoundingBox(int(results.xyxy[0][0][0]), int(results.xyxy[0][0][1]), int(results.xyxy[0][0][2]), int(results.xyxy[0][0][3]))

def model(input):
    scaled_inputs = torch.tan(torch.deg2rad(angle)) * torch.from_numpy(input)
    numerator = torch.from_numpy(9.81 * np.square(input))
    denominator = 2 * (velocity ** 2) * (torch.cos(torch.deg2rad(angle)) ** 2)
    return scaled_inputs - (numerator / denominator)

def plotTrajectory(coordinates, filename):
    outputs = model(np.arange(500))
    outputs = outputs.detach().numpy()
    # plt.scatter(coordinates[:, 0], coordinates[:,1])
    plt.plot(np.arange(500), outputs)
    plt.savefig(filename)

def fitTrajectoryCurve(coordinates, learning_rate, epochs):
    # fit trajectory curve
    optimizer = optim.Adam([angle, velocity], lr=learning_rate)
    loss_fn = nn.MSELoss()

    for i in range(epochs):
        optimizer.zero_grad()
        output = model(coordinates[:,0])
        loss = loss_fn(output, torch.from_numpy(coordinates[:,1]))
        loss.backward()
        optimizer.step()

cap = cv.VideoCapture('../basketball_clips/Alki-1/Micah1-Made.mov')

background_subtract = cv.createBackgroundSubtractorMOG2(500,1000,True)
useFrame = True
moving_pixel_locations = []
num_data_pts = 0
person_bbox = BoundingBox()
hoop_bbox = BoundingBox()
shouldFlip = False
isFirstFrame = True
enteredHoop = False
shotMade = False

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't read frame")
        break
    
    frame_width = np.shape(frame)[1]
    frame_height = np.shape(frame)[0]

    # only in first frame
    if isFirstFrame:
        isFirstFrame = False
        cv.imwrite("first_frame.jpg", frame)
        
        # person detect
        person_bbox = locatePerson("first_frame.jpg")
        # hoop detect
        hoop_bbox = detect("first_frame.jpg", "./weights/best.pt", 544, 0.1)

        person_bbox_center_x = (person_bbox.x1 + person_bbox.x2) / 2
        hoop_bbox_center_x = (hoop_bbox.x1 + hoop_bbox.x2) / 2

        if person_bbox_center_x > hoop_bbox_center_x:
            shouldFlip = True
            person_bbox = flipBoundingBox(person_bbox, frame_width)
            hoop_bbox = flipBoundingBox(hoop_bbox, frame_width)
 
    if shouldFlip:
        frame = cv.flip(frame, 1)
    
    # get motion mask
    motion_mask = background_subtract.apply(frame)

    top_hoop_bbox = motion_mask[hoop_bbox.y1 - 10, hoop_bbox.x1:hoop_bbox.x2]
    top_hoop_bbox_motion = np.argwhere(top_hoop_bbox == 255)
    if len(top_hoop_bbox_motion) > (hoop_bbox.getWidth() / 5):
        enteredHoop = True

    if enteredHoop:
        bottom_hoop_bbox = motion_mask[hoop_bbox.y2, hoop_bbox.x1:hoop_bbox.x2]
        bottom_hoop_bbox_motion = np.argwhere(bottom_hoop_bbox == 255)
        if len(bottom_hoop_bbox_motion) > (hoop_bbox.getWidth() / 5):
            shotMade = True

    # crop to just contain shot
    motion_crop = motion_mask[0:person_bbox.y1, person_bbox.x2:hoop_bbox.x1]
    
    # resize for faster computations
    resized_motion_crop = cv.resize(motion_crop, (0,0), fx=0.5, fy=0.5)

    motion_crop_height = np.shape(resized_motion_crop)[0]
    motion_crop_width = np.shape(resized_motion_crop)[1]

    # find indices of moving pixel 
    new_pixel_locations = np.argwhere(resized_motion_crop == 255)
    moving_pixel_locations.append(new_pixel_locations)
    num_data_pts += np.shape(new_pixel_locations)[0]
    
    cv.imshow('motion_mask', resized_motion_crop)
    cv.imshow('hoop mask', motion_mask[(hoop_bbox.y1 - 10):hoop_bbox.y2, hoop_bbox.x1:hoop_bbox.x2])
    if cv.waitKey(15) & 0xFF == ord('q'):
        break

coordinates = pixelLocationsToCoordinates(moving_pixel_locations, num_data_pts, motion_crop_height)
plotMovingPixels(coordinates)

fitTrajectoryCurve(coordinates, 1, 1000)
plotTrajectory(coordinates, "after-fit.jpg")
print("Angle: " + str(angle))

print("shot made: " + str(shotMade))

cap.release()
cv.destroyAllWindows()