import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import os
import cv2
import torch
import sys
from ament_index_python.packages import get_package_share_directory
import os


# sys.path.append('src/GroundingDINO')
# sys.path.append('src/yolov5')

from groundingdino.util.inference import load_model, load_image, predict, annotate
from pathlib import Path

class InitialDataProcessingNode(Node):
    def __init__(self):
        super().__init__('initial_data_processing_node')

        # ROS2 publishers and subscribers
        self.image_subscriber = self.create_subscription(
            Image, '/zed2i/zed_node/rgb/image_rect_color', self.image_callback, 10)
        self.annotated_image_publisher = self.create_publisher(Image, '/processed/annotated_image', 10)
        self.bbox_publisher = self.create_publisher(String, '/processed/yolo_labels', 10)
        # self.device = torch.device("cpu")
        self.bridge = CvBridge()

        # Load Grounding DINO model
 # Adjust with your weights path
        package_share_directory = get_package_share_directory('my_package')

        config_path = os.path.join(os.path.dirname(__file__), 'GroundingDINO_SwinT_OGC.py')
        weights_path = os.path.join(package_share_directory, 'config', 'groundingdino_swint_ogc.pth')

        self.model = load_model(config_path, weights_path)

        # Prediction settings
        self.text_prompt = "pallet, ground"
        self.box_threshold = 0.22
        self.text_threshold = 0.09

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img_height, img_width = cv_image.shape[:2]

        # Convert cv_image from BGR to RGB as required by the model
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Run Grounding DINO predictions directly
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_rgb,  # Pass the image directly as RGB
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )

        # Annotate image
        annotated_image = annotate(image_source=cv_image, boxes=boxes, logits=logits, phrases=phrases)
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Convert annotated image to ROS2 format and publish
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image_bgr, encoding="bgr8")
        self.annotated_image_publisher.publish(annotated_msg)

        # Generate YOLO labels and publish
        yolo_labels = []
        for box, label in zip(boxes, phrases):
            x_min, y_min, x_max, y_max = box

            # Clip bounding box coordinates within image bounds
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(img_width - 1, x_max), min(img_height - 1, y_max)

            # Convert to YOLO format (x_center, y_center, width, height) in normalized coordinates
            x_center = ((x_min + x_max) / 2)
            y_center = ((y_min + y_max) / 2)
            width = (x_max - x_min)
            height = (y_max - y_min)

            # Assign class index based on detected label
            class_index = 0 if "pallet" in label.lower() else 1 if "ground" in label.lower() else -1
            if class_index == -1:
                continue  # Skip unrecognized classes

            # Append to YOLO labels if bounding box is valid
            if width > 0 and height > 0:
                yolo_label = f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_labels.append(yolo_label)

    # Publish YOLO labels as a single string message
            self.bbox_publisher.publish(String(data="\n".join(yolo_labels)))


def main(args=None):
    rclpy.init(args=args)
    node = InitialDataProcessingNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
