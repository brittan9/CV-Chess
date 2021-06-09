import './App.css';
// Importing the Bootstrap CSS
import 'bootstrap/dist/css/bootstrap.min.css';
import trajectory from './data/trajectory.jpeg'
import shot_graph from './data/shot_graph.jpeg'

function App() {
  return (
    <div className="App">
        <h1 className="App-header">Nothing But Net</h1>
          <h5>CSE 455 Computer Vision - Final Project</h5>
          <h5>Team: Micah Wittaus and Brittan Robinett</h5>
          <section>
          <h2>Project Goals</h2>
            <p>
              Inspired by one of our shared hobbies and being stuck inside this year, our team wanted to use
              our newfound computer vision and machine learning skills to analyze video of basketball shots. 
              Our initial goal was to be able to take video with a mobile device of someone shooting hoops 
              and determine both the angle of their shot and whether it makes it into the hoop or not. We added some constraints
              to this problem by limiting the use case to hoops with a net, videos shot perpendicular to the player and the hoop, as well
              as requiring stable footage. Knowing the shot angle is helpful for players of all levels of experience, from beginner to professional.
              Many people have analyzed the physics behind a basketball shot and determined the "optimal" angle for release
              and entry - one source reported between 35 and 55 degrees depending on player height and shot distance. 
              While it's not a one-fits-all approach, the release angle does directly impact the outcome, and thus it's good to know 
              when analzying your performance and consistency. 
            </p>
            </section>
            <h2>Implementation</h2>
            <h4>Data Collection and Prep</h4>
              <p>
                In order to detect the basketball hoop in our video, we needed to train a neural net by providing as many images of hoops
                as we could reasonably find. Our first approach was going out and taking photos of around 10 hoops near us - but we quickly realized
                that it wouldn't be enough in terms of quantity and variety. Because of this, we padded out the rest of our dataset with
                images found online. We got a variety of images containing empty courts or people playing, indoor or outdoor courts, and a variety of 
                different looking basketball hoops and nets (we required there to be a net for our purposes). Finally, we used Roboflow to label bounding boxes, 
                augment, split into train/test/validation, and export in the correct format for training. Our final dataset contained 813 images (after augmentation).
              </p>
              <p>
                For testing our application, we took several videos on mobile devices at 3 different locations on different days (one day was shot handheld and the other day we used a tripod). 
                These videos also included people of different heights. Because of COVID, we were unable to take our own video at any indoor courts
                but based on our testing we believe that our application would likely perform better indoors because outdoors we had a great deal of background
                movement from cars, trees, etc. skewing our algorithm's perception of where the ball is. 
              </p>
            <h4>Object Detection: Hoop and Person</h4>
              <p>
                Our team chose to use Yolov5 for object detection because YOLO object detection methods are fast and accurate compared to other methods
                like Fast R-CNN. Yolov5 offers compatability with PyTorch and Colab, and has a "small" version that seemed perfect for our purposes since
                it runs extremely fast (we were considering running on a mobile device but didn't get to this due to time).
              </p>
              <p>
                Because Yolov5 trains in Google Colab, we trained once using our custom dataset and saved the best performing weights over 60 epochs. 
                The best mAP we achieved was around 0.9 for a confidence of 50% and above. The custom dataset tutorial we referenced is linked in the 
                References section. For detecting the person shooting in the image, we use a version of Yolov5 pretrained on the COCO dataset loaded 
                from PyTorch Hub, and constrained it only to detect Class 0, or "Person". 
              </p>
              <p>
                If the algorithm detects multiple hoops or multiple people, it chooses the one it detects with highest confidence. 
              </p>
            <h4>Shot Analysis: Trajectory Fitting and Shot Classifcation</h4>
              <p>
                Using the bounding box of the person and the hoop, we are able to locate the area that would contain the arc of the shot
                which we then looked for moving pixels within in order to isolate the movement of the ball. We use an OpenCV foreground/background 
                segmentation method that locates all moving pixels in a frame within a threshold to accomplish this. Because we wanted to get the entire
                shot over multiple frames of video, we bulit up a list of moving pixel locations and then had to convert them to coordinate locations on a
                cartesian xy-plane. 
              </p>
              <img className="photo" src={shot_graph} alt="shot graph of moving pixels"/>
              <p>
                After we gathered all the moving pixel coordinates, we fit a trajectory curve to our data using an Adam optimizer to get the angle 
                and velocity of our shot. 
              </p>
              <img className="photo" src={trajectory} alt="trajectory curve with variables"/>
              <p>
                To determine whether or not the shot made it in, we check for moving pixels in a region above the rim of the hoop, and if there is enough
                movement in that region, we start checking for moving pixels in a region below the net (both need to be passed through for the shot to count).
                This is important because we may detect moving pixels above, next to, behind, or generally near the rim/net without it actually passing 
                through the net. 
              </p>
              <h4>Displaying Analysis</h4>
              <p>
                In order to display the results of the shot analysis to the user we chose to make a text file, called "shot_stats.txt", containing the shot number,
                the angle of the shot, and whether it was made or not for each shot they passed into the program. At the bottom of the 
                of the text file we display the total number of shots made out of the the total number of shot attempts, the percentage 
                of shots made, and the average shot angle. We also save new video files with the same names as the previous ones with the 
                exception of "_analysis.avi" tacked on to the end of them that contain the original video with the shot arc overlayed on it 
                (the shot arc is red for shots detected as misses and green otherwise).
              </p>
            <h2>Results</h2>

            <h2>Reflection</h2>
              <h4>Alternative Approaches</h4>
                <p>
                One problem we encountered was background movement introducing noise into the coordinate data we used to fit our trajectory 
                curve; this problem was especially apparent on the footage we shot handheld. An alternative approach that would circumvent
                this problem would be to track the basketball throughout the frames in the video using yolov5 and take the coordinates from
                the center of its bounding box. This approach has two methods of implementation; the first would be to use a version of yolov5
                pre-trained on the COCO dataset and use the class sports ball to locate the basketball, the second would be to create a custom
                dataset of various types of basketballs and train yolov5 on it in order to do the detection. We ended up foregoing these approaches
                for a few reasons. Firstly, if we used the pretrained yolov5 we didn't think that sportsball represented what we were trying to detect
                as much as a custom dataset could. Secondly, if we chose to create a custom dataset we didn't think we would have enough data to create
                a representive group of basketballs of different styles in various environments (different lighting, position, etc). Lastly,
                one of our biggest concerns was how well the basketball detection would work as the ball passes through the hoop since this is an
                integral part of detecting whether a shot went in or not.
                </p>
              <h4>Future Work</h4>
                <p>
                There is a lot of functionality that could be built onto this project and our team plans to continue work on it when we can!
                In the future we might implement a hybrid of the approach discussed in "Alternative Approaches" and what we implemented, where we use
                the basketball tracking until the ball reaches the hoop then use motion tracking to see if the shot was made. Another thing 
                we would want to be able to do is take video from any location on the court instead of needing to be 
                perpendicular to the shot - this could be achieved using a translation and mapping to 2D space, or alternatively
                could be done by using a 3D trajectory formula if you can determine the precise player location on the court (potentially using homographies).
                Additionally, for usability we really want users to be able to have shot analysis run directly on their mobile device instaed of needing to
                upload to a laptop or home computer. Lastly, another functionality we would like to implement is the ability for our algorithm to
                trim videos automatically by recognizing what a shot looks like using binary classifcation.
                </p>
              <h2>Links and References</h2>
                <p>Camera-based Basketball Scoring Detection Using CNN: <link>http://www.ijac.net/fileIJAC/journal/article/ijac/2021/2/PDF/IJAC-2020-05-119.pdf</link></p>
                <p>Information on shot angles: <link>https://www.noahbasketball.com/blog/is-a-higher-arc-really-better</link></p>
                <p>Shooting Hoops with Keras and Tensorflow - Zack Akil <link>https://www.youtube.com/watch?v=S9PcPbtTcPc&t=811s</link></p>
                <p></p>
    </div>
  );
}

export default App;
