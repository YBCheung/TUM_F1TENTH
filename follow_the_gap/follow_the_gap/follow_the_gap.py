import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from scipy.ndimage import uniform_filter1d
import numpy as np
import math
from scipy.stats import norm 

class FollowTheGapNode(Node):
    def __init__(self):
        # Initialize the node with the name 'follow_the_gap_node'
        super().__init__('follow_the_gap_node')

        # Declare parameters with default values
        self.declare_parameter('bubble_radius', 0.1)
        self.declare_parameter('smoothing_filter_size', 3)
        self.declare_parameter('truncated_coverage_angle', 180.0)
        self.declare_parameter('max_accepted_distance', 10.0)
        self.declare_parameter('error_based_velocities.low', 2.0)
        self.declare_parameter('error_based_velocities.medium', 1.0)
        self.declare_parameter('error_based_velocities.high', 0.5)
        self.declare_parameter('steering_angle_reactivity', 0.18)

        # Setup subscription to the lidar scans and publisher for the drive command
        self.lidar_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, 'drive', 10)

        # Flag to check if the truncated indices have been computed
        self.truncated = False

    def apply_smoothing_filter(self, input_vector, smoothing_filter_size):
        # Apply a uniform 1D filter to smooth the input vector, which helps to mitigate noise in the lidar data
        return uniform_filter1d(input_vector, size=smoothing_filter_size, mode='nearest')

    def truncated_start_and_end_indices(self, scan_msg, truncation_angle_coverage):
        # Calculate start and end indices for the truncated view based on the desired coverage angle
        truncated_range_size = int(
            truncation_angle_coverage / (scan_msg.angle_max - scan_msg.angle_min) * len(scan_msg.ranges))
        start_index = len(scan_msg.ranges) // 2 - truncated_range_size // 2
        end_index = len(scan_msg.ranges) // 2 + truncated_range_size // 2
        return start_index, end_index

    def minimum_element_index(self, input_vector):
        # Find the index of the minimum element in the input vector
        return int(np.argmin(input_vector))

    def find_largest_nonzero_sequence(self, input_vector):
        # Find the start indices of the largest sequence of non-zero values in the input vector
            
        # TASK 1: CALCULATE THE MAXIMUM GAP FROM THE LIDAR DATA
        # Hints:
        # - The lidar data array represents distances measured at different angles.
        # - Your task is to find the largest sequence of consecutive non-zero distances, which corresponds to the largest gap.
        # - Iterate through the 'filtered_ranges' array (the input_vector) to find the longest sequence of non-zero values.
        # - Use a loop and conditional statements to keep track of the start and end indices of the current sequence.
        # - If a value is greater than a threshold (0.1), it is part of a gap.
        # - If a value is zero or less than the threshold, it signifies the end of a current gap.
        # - Keep track of the maximum gap found so far.

        # Replace this section within the for loop with your code to find the 'max_start' and 'max_gap' values of the largest gap
        
        # Parameters
        window_len = 100

        # Variables
        seq_sum = 0
        start_point = -1
        current_len = 0

        max_win = [540, 0]
        sequences = []

        for i, dis in enumerate(input_vector):
            if dis > 0.1:
                if current_len == 0:
                    # start of a new window
                    start_point = i
                    current_len = 1
                    seq_sum = input_vector[i]
                
                elif current_len < 100:
                    # window growing
                    seq_sum += input_vector[i]
                    current_len += 1

                elif current_len == window_len:
                    # full window slide
                    seq_sum = seq_sum + input_vector[i] - input_vector[start_point]
                    start_point += 1
                    if seq_sum > max_win[1]:
                        max_win = [start_point, seq_sum]
            elif current_len > 0:
                current_len = 0
                seq_sum = 0
        return max_win[0], max_win[0] + window_len


    def zero_out_safety_bubble(self, input_vector, center_index, bubble_radius):
        # Zero out the elements within the 'bubble_radius' around the 'center_index'
        # This creates a 'bubble' around the closest obstacle where no valid paths can exist
        center_point_distance = input_vector[center_index]
        input_vector[center_index] = 0.0

        # Expand the bubble to the right
        current_index = center_index
        while current_index < len(input_vector) - 1 and input_vector[
            current_index + 1] < center_point_distance + bubble_radius:
            current_index += 1
            input_vector[current_index] = 0.0

        # Expand the bubble to the left
        current_index = center_index
        while current_index > 0 and input_vector[current_index - 1] < center_point_distance + bubble_radius:
            current_index -= 1
            input_vector[current_index] = 0.0

    def preprocess_lidar_scan(self, scan_msg):
        # Preprocess the lidar scan data
        # Convert NaNs to 0 for processing and cap the max range to a set value
        ranges = np.array(scan_msg.ranges)
        ranges[np.isnan(ranges)] = 0.0
        ranges[ranges > self.get_parameter('max_accepted_distance').get_parameter_value().double_value] = \
            self.get_parameter('max_accepted_distance').get_parameter_value().double_value
        # Apply the smoothing filter
        return self.apply_smoothing_filter(ranges,
                                      self.get_parameter('smoothing_filter_size').get_parameter_value().integer_value)

    def get_best_point(self, filtered_ranges, start_index, end_index):   # zyb_TODO: use sigmoid filter!
        # Determine the best point to aim for within the largest gap
        return (start_index + end_index) // 2
    
    def sigmoid_angle(x, threshold):
        return 1 / (1 + math.exp(-x + threshold))
    def norm_speed(x):
        return norm.pdf(x, 0, 0.7) 
    
    def get_steering_angle_from_range_index(self, scan_msg, best_point_index, closest_index, closest_value):
        # Convert the index of the best point in the range array to a steering angle
        increment = scan_msg.angle_increment
        num_ranges = len(scan_msg.ranges)
        mid_point = num_ranges // 2

        # TASK 2A: CALCULATE THE STEERING ANGLE BASED ON THE MAXIMUM GAP
        # Hints:
        # - You need to calculate the angle to the center of the maximum gap.
        # - The lidar data is an array where each element corresponds to the distance measured at a specific angle increment.
        # - The index of the array represents the angular position relative to the front of the vehicle, with the center being the forward direction.
        # - Use the following variables:
        #   'best_point_index' - the index of the epoint to aim for within the maximum gap
        #   'mid_point' - the index that represents straight ahead
        #   'increment' - the angular distance between each lidar measurement
        # - The angle to the best point is the difference between 'best_point_index' and 'mid_point' times 'increment'.
        # - Remember to adjust the sign of the angle depending on whether the best point is to the left or right of center.
        # - If you need to read in a parameter, have a look at other lines in this codes to find the syntax.

        # Insert your code here to calculate 'best_point_steering_angle'
        best_point_steering_angle = increment * (best_point_index - mid_point) 
        
        # TASK 2B: CALCULATE THE COMPENSATED STEERING ANGLE
        # Hints:
        # - The compensated steering angle adjusts the basic steering angle based on the proximity of the closest obstacle.
        # - This helps the vehicle to steer more aggressively when obstacles are close and more smoothly when they are distant.
        # - Use the following variables:
        #   'best_point_steering_angle' - the previously calculated steering angle
        #   'steering_angle_reactivity' - a parameter for tuning the reactivity of the steering to the distance of the closest obstacle
        #   'closest_value' - the distance to the closest obstacle
        # - The compensation is calculated as the steering angle times the reactivity divided by the closest obstacle distance.
        # - Use the 'np.clip' function to limit the compensated steering angle within the range of -1.57 to 1.57 radians.
        # - Understand that this formula is heuristic and can be adjusted based on testing and the specific dynamics of your vehicle.

        # Insert your code here to calculate 'distance_compensated_steering_angle'
        steering_angle_reactivity = 0.08  # zyb: adjust parameter
        distance_compensated_steering_angle = best_point_steering_angle \
            - np.sign(closest_index - mid_point) * steering_angle_reactivity / closest_value
        
        print(f"A1: {best_point_steering_angle:.2f},\t A2: {- np.sign(closest_index - mid_point) * steering_angle_reactivity / closest_value:.2f},\t A3: {distance_compensated_steering_angle:.2f},\t dis: {closest_value:.2f},\t best: {best_point_index - mid_point},\t idx: {closest_index - mid_point}")

        distance_compensated_steering_angle = np.clip(distance_compensated_steering_angle, -1.57, 1.57)

        return distance_compensated_steering_angle

    def scan_callback(self, scan_msg):
        # Callback function for processing lidar scans
        # First time this is run, we calculate the truncated indices based on the desired angle coverage
        if not self.truncated:
            truncated_indices = self.truncated_start_and_end_indices(scan_msg, self.get_parameter(
                'truncated_coverage_angle').get_parameter_value().double_value)
            self.get_logger().info(f"Truncated Indices: {truncated_indices}")
            self.truncated_start_index, self.truncated_end_index = truncated_indices
            self.truncated = True

        # Process the lidar scan data
        filtered_ranges = self.preprocess_lidar_scan(scan_msg)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))
        
        # Find the closest obstacle
        closest_index = self.minimum_element_index(filtered_ranges)
        closest_range = filtered_ranges[closest_index]

        # Zero out the safety bubble around the closest obstacle
        self.zero_out_safety_bubble(filtered_ranges, closest_index,
                               self.get_parameter('bubble_radius').get_parameter_value().double_value)

        # Find the largest gap in the scan data
        start_index, end_index = self.find_largest_nonzero_sequence(filtered_ranges)
        # Determine the best point within that gap to aim for
        best_point_index = self.get_best_point(filtered_ranges, start_index, end_index)

        # Get the steering angle that directs the car towards the best point
        steering_angle = self.get_steering_angle_from_range_index(scan_msg, best_point_index, closest_index, closest_range)

        # Prepare the drive message with the steering angle and corresponding speed
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'laser'
        drive_msg.drive.steering_angle = steering_angle


        # TASK 3: SET THE CAR'S VELOCITY DEPENDENT ON THE STEERING ANGLE
        # Hints:
        # - The car's velocity should be inversely proportional to the steering angle. The sharper the turn, the slower the car should move.
        # - You need to determine the appropriate speed based on the magnitude of the steering angle.
        # - Use conditional statements to set different speeds for different ranges of the steering angle.
        # - Three speed parameters are provided: 'high', 'medium', and 'low'. Assign them based on the steering angle's absolute value.
        # - The provided angle thresholds (0.349 radians for 'high' speed and 0.174 radians for 'medium' speed) represent the steering angle beyond which the speed should be reduced.
        # - Translate the above logic into an 'if-elif-else' structure that sets the drive_msg speed based on the steering angle.

        # Replace this section with your code that sets the 'drive_msg.drive.speed' based on the 'steering_angle'
        
        if abs(steering_angle) > 0.349:    # adjust speed according to angle and distance.
            drive_msg.drive.speed = 0.8
        elif abs(steering_angle) > 0.174:
            drive_msg.drive.speed = 1.0
        else:
            drive_msg.drive.speed = 2.0
        

        # Publish the drive command
        self.drive_pub.publish(drive_msg)

def main(args=None):
    # Main function to initialize the ROS node and spin
    rclpy.init(args=args)
    node = FollowTheGapNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    # Entry point for the script
    main()