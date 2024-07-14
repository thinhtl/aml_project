import cv2  # Import OpenCV library for image processing

from ultralytics.utils.checks import check_imshow  # Import function to check if imshow is supported
from ultralytics.utils.plotting import Annotator  # Import Annotator for drawing and plotting annotations on images

class WorkoutMonitor:
    """
    WorkoutMonitor class to monitor workout exercises and count repetitions using pose estimation.

    Attributes:
        kpts_to_check (list): List of keypoints to check for angle estimation.
        line_thickness (int): Thickness of lines drawn on the image.
        view_img (bool): Flag to display image during processing.
        pose_up_angle (float): Angle threshold for the "up" position.
        pose_down_angle (float): Angle threshold for the "down" position.
        pose_type (str): Type of exercise (e.g., "pullup", "pushup").
    """

    def __init__(
        self,
        kpts_to_check,
        line_thickness=2,
        view_img=False,
        pose_up_angle=145.0,
        pose_down_angle=90.0,
        pose_type="pullup",
    ):
        """
        Initialize the WorkoutMonitor class.

        Parameters:
            kpts_to_check (list): List of keypoints to check for angle estimation.
            line_thickness (int): Thickness of lines drawn on the image. Default is 2.
            view_img (bool): Flag to display image during processing. Default is False.
            pose_up_angle (float): Angle threshold for the "up" position. Default is 145.0.
            pose_down_angle (float): Angle threshold for the "down" position. Default is 90.0.
            pose_type (str): Type of exercise (e.g., "pullup", "pushup"). Default is "pullup".
        """
        self.im0 = None  # Initialize image variable
        self.tf = line_thickness  # Set line thickness for drawing

        self.keypoints = None  # Initialize keypoints variable
        self.poseup_angle = pose_up_angle  # Set the angle threshold for the "up" position
        self.posedown_angle = pose_down_angle  # Set the angle threshold for the "down" position
        self.threshold = 0.001  # Set a threshold value

        self.angle = None  # Initialize angle list
        self.count = None  # Initialize count list
        self.stage = None  # Initialize stage list
        self.pose_type = pose_type  # Set the type of exercise
        self.kpts_to_check = kpts_to_check  # Set the keypoints to be checked

        self.view_img = view_img  # Set whether to view images
        self.annotator = None  # Initialize annotator

        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)  # Check if imshow is supported in the current environment
        self.count = []  # Initialize count list
        self.angle = []  # Initialize angle list
        self.stage = []  # Initialize stage list

    def start_counting(self, im0, results):
        """
        Start counting workout repetitions based on pose estimation.

        Parameters:
            im0 (numpy.ndarray): The current image/frame to be processed.
            results (list): List containing keypoints data from pose estimation model.

        Returns:
            numpy.ndarray: The annotated image/frame.
        """
        self.im0 = im0  # Set the current image

        if not len(results[0]):
            return self.im0  # Return the image if no results are found

        if len(results[0]) > len(self.count):
            new_human = len(results[0]) - len(self.count)  # Calculate the number of new humans detected
            self.count += [0] * new_human  # Initialize count for new humans
            self.angle += [0] * new_human  # Initialize angle for new humans
            self.stage += ["-"] * new_human  # Initialize stage for new humans

        self.keypoints = results[0].keypoints.data  # Get keypoints data from results
        self.annotator = Annotator(im0, line_width=self.tf)  # Initialize annotator with the current image

        for ind, k in enumerate(reversed(self.keypoints)):
            if self.pose_type in {"pushup", "pullup", "abworkout", "squat"}:
                # Estimate the pose angle based on keypoints
                self.angle[ind] = self.annotator.estimate_pose_angle(
                    k[int(self.kpts_to_check[0])].cpu(),
                    k[int(self.kpts_to_check[1])].cpu(),
                    k[int(self.kpts_to_check[2])].cpu(),
                )
                # Draw specific points on the image
                self.im0 = self.annotator.draw_specific_points(k, self.kpts_to_check, shape=(640, 640), radius=10)

                if self.pose_type in {"abworkout", "pullup"}:
                    if self.angle[ind] > self.poseup_angle:
                        self.stage[ind] = "down"  # Set stage to "down" if angle is greater than the up angle
                    if self.angle[ind] < self.posedown_angle and self.stage[ind] == "down":
                        self.stage[ind] = "up"  # Set stage to "up" if angle is less than the down angle
                        self.count[ind] += 1  # Increment count

                elif self.pose_type in {"pushup", "squat"}:
                    if self.angle[ind] > self.poseup_angle:
                        self.stage[ind] = "up"  # Set stage to "up" if angle is greater than the up angle
                    if self.angle[ind] < self.posedown_angle and self.stage[ind] == "up":
                        self.stage[ind] = "down"  # Set stage to "down" if angle is less than the down angle
                        self.count[ind] += 1  # Increment count

                # Plot angle, count, and stage on the image
                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],
                    count_text=self.count[ind],
                    stage_text=self.stage[ind],
                    center_kpt=k[int(self.kpts_to_check[1])],
                )

            # Draw keypoints on the image
            self.annotator.kpts(k, shape=(640, 640), radius=1, kpt_line=True)

        if self.env_check and self.view_img:
            cv2.imshow("Workout monitor", self.im0)  # Display the image
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return  # Exit if 'q' is pressed

        return self.im0  # Return the annotated image


if __name__ == "__main__":
    kpts_to_check = [0, 1, 2]  # example keypoints
    workout_monitor = WorkoutMonitor(kpts_to_check)  # Initialize WorkoutMonitor with example keypoints
