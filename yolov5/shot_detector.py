import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch._C import dtype
import torch.optim as optim
import torch.nn as nn
import sys
from bounding_box import BoundingBox
from detect import detect

# globals for ML curve fitting
angle = nn.Parameter(torch.tensor(45.0), requires_grad=True)
velocity = nn.Parameter(torch.randint(40, 51, (1,), dtype=torch.float64), requires_grad=True)

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

# For the use of flipping OpenCV coordinates across the y-axis
def flipCoordinates(coordinates, frame_width):
    for coordinate in coordinates:
        coordinate[0] = frame_width - coordinate[0]
    return coordinate

def locatePerson(img_path):
    # person detect
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.conf = 0.75
    model.classes = [0]
    results = model(img_path, size=544)
    highest_confidence_seen = 0
    person_bbox = BoundingBox()
    for people in results.xyxy[0]:
        if people[4] > highest_confidence_seen:
            highest_confidence_seen = people[4]
            person_bbox = BoundingBox(int(people[0]), int(people[1]), int(people[2]), int(people[3]))
    return person_bbox

def model(input):
    scaled_inputs = torch.tan(torch.deg2rad(angle)) * torch.from_numpy(input)
    numerator = torch.from_numpy(9.81 * np.square(input))
    denominator = 2 * (velocity ** 2) * (torch.cos(torch.deg2rad(angle)) ** 2)
    return scaled_inputs - (numerator / denominator)

def plotTrajectory(last_x, filename):
    outputs = model(np.arange(last_x))
    outputs = outputs.detach().numpy()
    # plt.scatter(coordinates[:, 0], coordinates[:,1])
    plt.plot(np.arange(last_x), outputs, c='r')
    plt.savefig(filename)
    plt.clf()

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

def getTrajectoryInCoordinates(last_x):
    outputs = model(np.arange(last_x))
    outputs = outputs.detach().numpy()
    trajectory_coordinates = np.empty((len(outputs), 2), dtype=np.int32)
    for i, predicted_y in enumerate(outputs):
        trajectory_coordinates[i, 0] = i
        trajectory_coordinates[i, 1] = predicted_y
    return trajectory_coordinates

def coordinatesToOriginalPixelLocations(trajectory_coordinates, height, x_offset):
    for coordinates in trajectory_coordinates:
        coordinates[1] = height - coordinates[1]
        coordinates[0] *= 2
        coordinates[1] *= 2
        coordinates[0] += x_offset

    return trajectory_coordinates

def main():
    file = open("shot_stats.txt", "w")
    num_shots_made = 0
    sum_of_shot_angles = 0
    for shot_num, arg in enumerate(sys.argv[1:]):
        cap = cv.VideoCapture(arg)

        shouldUseArcData = True
        background_subtract = cv.createBackgroundSubtractorMOG2(500,800,True)
        moving_pixel_locations = []
        num_data_pts = 0
        person_bbox = BoundingBox()
        hoop_bbox = BoundingBox()
        shouldFlip = False
        isFirstFrame = True
        enteredHoop = False
        shotMade = False
        motion_crop_height = 0
        last_x = 0
        original_frame_width = 0
        original_frame_height = 0

        while cap.isOpened():
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break
            
            frame_width = np.shape(frame)[1]
            original_frame_height = np.shape(frame)[0]
            original_frame_width = np.shape(frame)[1]

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

            top_hoop_bbox = motion_mask[hoop_bbox.y1 - 10:hoop_bbox.y1 + 10, hoop_bbox.x1:hoop_bbox.x2]
            top_hoop_bbox_motion = np.argwhere(top_hoop_bbox == 255)
            percentage_pixels_in_motion = len(top_hoop_bbox_motion) / (hoop_bbox.getWidth() * 20)
            if percentage_pixels_in_motion > 0.05:
                enteredHoop = True

            if enteredHoop:
                bottom_hoop_bbox = motion_mask[hoop_bbox.y2 - 10:hoop_bbox.y2 + 10, hoop_bbox.x1:hoop_bbox.x2]
                bottom_hoop_bbox_motion = np.argwhere(bottom_hoop_bbox == 255)
                percentage_pixels_in_motion = len(bottom_hoop_bbox_motion) / (hoop_bbox.getWidth() * 20)
                if percentage_pixels_in_motion > 0.1:
                    shotMade = True
                    shouldUseArcData = False
            # crop to just contain shot
            motion_crop = motion_mask[0:person_bbox.y1, person_bbox.x2:hoop_bbox.x1]
            
            # resize for faster computations
            resized_motion_crop = cv.resize(motion_crop, (0,0), fx=0.5, fy=0.5)

            motion_crop_height = np.shape(resized_motion_crop)[0]

            # find indices of moving pixel
            if shouldUseArcData:
                new_pixel_locations = np.argwhere(resized_motion_crop == 255)
                moving_pixel_locations.append(new_pixel_locations)
                num_data_pts += np.shape(new_pixel_locations)[0]
            if len(new_pixel_locations) != 0:
                last_x = max(np.amax(new_pixel_locations, axis=0)[1], last_x)

            if cv.waitKey(15) & 0xFF == ord('q'):
                break
        cap.release()
        cap = cv.VideoCapture(arg)
        coordinates = pixelLocationsToCoordinates(moving_pixel_locations, num_data_pts, motion_crop_height)
        plotMovingPixels(coordinates)

        fitTrajectoryCurve(coordinates, 0.1, 1000)
        plotTrajectory(last_x, str(arg[:len(arg) - 4]) + "_fit.jpg")
        trajectory_coordinates = getTrajectoryInCoordinates(last_x)
        arc_points = coordinatesToOriginalPixelLocations(trajectory_coordinates, motion_crop_height, person_bbox.x2)
        arc_points.reshape((-1, 1, 2))
        arc_color = (0, 0, 0)
        if shotMade:
            arc_color = (0, 255, 0)
        else:
            arc_color = (0, 0, 255)
        arc_thickness = 8
        isClosed = False
        sum_of_shot_angles += angle.item()
        if shotMade:
            num_shots_made += 1
        file.write("Shot #" + str(shot_num + 1) + "\nShot angle: " + str(angle.item()) + "\nShot made it: " + str(shotMade) + "\n\n")
        result = cv.VideoWriter(arg[:len(arg) - 4] + '_analysis.avi', cv.VideoWriter_fourcc(*'MJPG'), 30, (original_frame_width, original_frame_height))
        while cap.isOpened():
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break

            if shouldFlip:
                #flipCoordinates(trajectory_coordinates, motion_crop_width)
                frame = cv.flip(frame, 1)
            frame = cv.polylines(frame, [arc_points], 
                            isClosed, arc_color, 
                            arc_thickness)
            result.write(frame)
            if cv.waitKey(15) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
    file.write("Number of shots made: " + str(num_shots_made) + "/" + str(len(sys.argv) - 1))
    file.write("\nPercentage of shots made: " + str(100 * (num_shots_made / (len(sys.argv) - 1))))
    file.write("\nAverage shot angle: " + str(sum_of_shot_angles / len(sys.argv) - 1))
    file.close()

if __name__ == '__main__':
    main()