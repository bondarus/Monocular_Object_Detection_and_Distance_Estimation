import cv2
import imutils
import math
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

def calculate_distance(focal_length, real_width, perceived_width):
    if perceived_width > 0:
        return (real_width * focal_length) / perceived_width
    return None

# Function to calculate the distance between two objects using the cosine rule
def calculate_object_distance(d1, d2, angle_between):
    angle_radians = math.radians(angle_between)
    return math.sqrt(d1**2 + d2**2 - 2 * d1 * d2 * math.cos(angle_radians))

# Sample function to compute the angle and distance between bicycle and vehicle
def compute_bicycle_vehicle_distance(bicycle_box, vehicle_box, d_bicycle, d_vehicle, fov, frame_width):
    # Calculate the centre x-coordinates of the bounding boxes
    x1 = (bicycle_box[0] + bicycle_box[2]) / 2  # Centre x of bicycle
    x2 = (vehicle_box[0] + vehicle_box[2]) / 2  # Centre x of vehicle

    # Calculate the angles for both objects relative to the camera's FOV
    angle_bicycle = (x1 / frame_width - 0.5) * fov
    angle_vehicle = (x2 / frame_width - 0.5) * fov
    angle_between = abs(angle_bicycle - angle_vehicle)

    # Calculate the distance between the objects
    distance_between = calculate_object_distance(d_bicycle, d_vehicle, angle_between)
    return distance_between

def display_info(frame, distance, perceived_width, frame_number):
    info_text = f"Frame: {frame_number} | Distance: {distance:.2f} mm | Width: {perceived_width} px"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Constants for distance calculation
FOCAL_LENGTH = 1900  # mm
BICYCLE_WIDTH = 500  # mm
PERSON_WIDTH = 500  # mm
LICENSE_PLATE_WIDTH = 520  # mm
FOV = 70


# Load YOLO models
model_person_bicycle = YOLO('yolov8n-seg.pt')  # For person and bicycle detection
model_license_plate = YOLO('license_plate_detector.pt')  # For license plate detection

# Define the class IDs for person and bicycles based on the model's training data
target_classes = [0, 1]  # Adjust based on your model's class indices

# Open the input video
input_video_path = './videos/video1.mp4'
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video file {input_video_path}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
output_video_path = './output/labeled_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_number = 0  # Initialize frame counter


# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame_number += 1  # Increment frame counter

    # Initialize annotator for drawing
    annotator = Annotator(frame)

    # Detect persons and bicycles
    results_person_bicycle = model_person_bicycle(frame)
    largest_bicycle_box = None
    largest_bicycle_area = 0
    largest_bicycle_distance = None

    largest_person_box = None
    largest_person_area = 0
    largest_person_distance = None
    person_boxes = []
    
    # Iterate bounding box over detected instances
    for result in results_person_bicycle:
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls)  # Class ID
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
                box_area = (x2 - x1) * (y2 - y1)

                if cls == 1:  # Bicycle class ID
                    if box_area > largest_bicycle_area:
                        largest_bicycle_area = box_area
                        largest_bicycle_box = (x1, y1, x2, y2)
                        perceived_width = x2 - x1
                        largest_bicycle_distance = calculate_distance(FOCAL_LENGTH, BICYCLE_WIDTH, perceived_width)
                elif cls == 0:  # Person class ID
                    person_boxes.append((x1, y1, x2, y2))

    # Draw the largest bicycle bounding box
    if largest_bicycle_box:
        x1, y1, x2, y2 = largest_bicycle_box
        annotator.box_label((x1, y1, x2, y2), f"Bicycle: {largest_bicycle_distance:.2f} mm", color=colors(1))

        for px1, py1, px2, py2 in person_boxes:
            ix1 = max(x1, px1)
            iy1 = max(y1, py1)
            ix2 = min(x2, px2)
            iy2 = min(y2, py2)

            if ix1 < ix2 and iy1 < iy2:  # Overlap exists
                largest_person_box = (px1, py1, px2, py2)
                perceived_width = px2 - px1
                distance = calculate_distance(FOCAL_LENGTH, PERSON_WIDTH, perceived_width)
                largest_person_distance = distance
                annotator.box_label((px1, py1, px2, py2), f"Person: {distance:.2f} mm", color=colors(0))

    # Detect license plates
    results_license_plate = model_license_plate(frame)
    largest_license_plate_box = None
    largest_license_plate_distance = None

    for result in results_license_plate:
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls)
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
                perceived_width = x2 - x1
                distance = calculate_distance(FOCAL_LENGTH, LICENSE_PLATE_WIDTH, perceived_width)

                # Track the largest license plate for distance calculation
                if largest_license_plate_distance is None or distance < largest_license_plate_distance:
                    largest_license_plate_distance = distance
                    largest_license_plate_box = (x1, y1, x2, y2)

                label = f"License Plate: {distance:.2f} mm"
                annotator.box_label((x1, y1, x2, y2), label, color=colors(cls))


    # Calculate distance between cyclist and vehicle using the cosine rule
    if largest_bicycle_distance and largest_license_plate_distance and largest_bicycle_box and largest_license_plate_box:
        # Use the closest x-axis edges for the bicycle and vehicle bounding boxes
        bicycle_x_edge = largest_person_box[2] if largest_person_box[2] < largest_license_plate_box[0] else largest_person_box[0]
        vehicle_x_edge = largest_license_plate_box[0] if largest_license_plate_box[0] > largest_person_box[2] else largest_license_plate_box[2]

        bicycle_edge_y = (largest_person_box[1] + largest_person_box[3]) / 2  # Average y-coordinate for the bicycle edge
        vehicle_edge_y = (largest_license_plate_box[1] + largest_license_plate_box[3]) / 2  # Average y-coordinate for the vehicle edge

        # Compute angles relative to the camera center for these edges
        bicycle_angle = (bicycle_x_edge / frame.shape[1] - 0.5) * FOV
        vehicle_angle = (vehicle_x_edge / frame.shape[1] - 0.5) * FOV
        angle_between = abs(bicycle_angle - vehicle_angle)

        # Calculate distance using the cosine rule function
        total_distance = calculate_object_distance(largest_person_distance, largest_license_plate_distance, angle_between)

        # Annotate the calculated distance on the frame
        cv2.putText(frame,f"Bicycle-Vehicle Distance: {total_distance:.2f} mm",(10, 60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),2,)

        # Define the edge points for visualization
        bicycle_edge = (int(bicycle_x_edge), int(bicycle_edge_y))
        vehicle_edge = (int(vehicle_x_edge), int(vehicle_edge_y))

        # Draw a line between the closest x-axis edges of the bicycle and the vehicle
        cv2.line(frame, bicycle_edge, vehicle_edge, (0, 255, 0), 2)  # Green line
        cv2.circle(frame, bicycle_edge, 5, (255, 0, 0), -1)  # Blue dot for bicycle edge
        cv2.circle(frame, vehicle_edge, 5, (0, 0, 255), -1)  # Red dot for vehicle edge


    # Get the annotated frame
    labeled_frame = annotator.result()

    # Write the frame to the output video
    out.write(labeled_frame)

    # Display the frame
    cv2.imshow('Bicycle and Vehicle Distance Estimation', labeled_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to: {output_video_path}")
