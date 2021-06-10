# CV-Nothing-But-Net

Hello! Welcome to our final project for CSE 455 - Computer Vision
Team: Brittan Robinett and Micah Witthaus

Project Website: http://brittan9.github.io/CV-Nothing-But-Net

Nothing But Net takes in trimmed videos of players making free throw basketball
shots before 1) finding the release angle, and 2) classifying the shot as made it in or not.

Important - For optimal results:
- Video contains only one person and only one hoop
- Video is shot from a spot perpendicular to the hoop and person
- Video is stable (handheld introduces noise, skewing the shot angle)

How To Use:

In your terminal, go into the yolov5 folder.
Run:
```
python3 shot_detector.py [space separated list of paths to trimmed shot videos]
```

Output: 

- In the location of the clips passed in, it will output .mp4 files containing the video with the
overlayed shot arc (green for made and red for miss)
- In yolov5 folder, a shot_stats.txt file will be created containing shot angles and whether it was made
or not for each shot, as well as overall shot statistics summarizing all shots together. 


