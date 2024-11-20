import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
import os
import cv2
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

print("YOLOv5 module imported successfully!")


# from yolov5.utils.general import LOGGER, check_version, colorstr, file_date, git_describe
# from your_package_name.yolov5.utils.torch_utils import select_device
# from your_package_name.yolov5.models.common import DetectMultiBackend
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device

print("DetectMultiBackend and select_device imported successfully!")



class YOLODetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')

        # ROS publishers and subscribers
        self.image_subscriber = self.create_subscription(
            Image, '/processed/annotated_image', self.image_callback, 10)
        self.label_subscriber = self.create_subscription(
            String, '/processed/yolo_labels', self.label_callback, 10)
        self.detection_publisher = self.create_publisher(String, '/processed/yolo_detections', 10)
        self.annotated_image_publisher = self.create_publisher(Image, '/processed/yolo_annotated_image', 10)

        # YOLO model setup
        self.bridge = CvBridge()
        weights_path = '/home/kunj/pallet_detect_ws/src/my_package/config/best.pt'  # Update path
        device = select_device('')  # Auto-select CPU or GPU
        self.model = DetectMultiBackend(weights_path, device=device)
        self.model.conf = 0.13  # Confidence threshold

        self.current_image = None

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.get_logger().info("Image received. Running inference...")
        self.process_detections()

    def label_callback(self, msg):
        # Capture YOLO label data (for reference, not used in this code)
        self.get_logger().info(f"Labels received: {msg.data}")

    def process_detections(self):
        if self.current_image is not None:
            # Prepare image for YOLOv5
            img = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

            # Run YOLO model inference
            results = self.model(img_tensor)
            detections = non_max_suppression(results, conf_thres=self.model.conf)

            # Annotate image with bounding boxes and labels
            annotated_image = self.current_image.copy()
            detection_msgs = []
            for det in detections[0]:  # Process detections
                x1, y1, x2, y2, conf, cls = det
                label = self.model.names[int(cls)]
                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label_text = f"{label} {conf:.2f}"
                cv2.putText(annotated_image, label_text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detection_msgs.append(f"{label} {conf:.2f} {x1:.0f} {y1:.0f} {x2:.0f} {y2:.0f}")

            # Publish annotated image
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
            self.annotated_image_publisher.publish(annotated_msg)

            # Publish detection results
            detection_message = "\n".join(detection_msgs)
            self.detection_publisher.publish(String(data=detection_message))

            self.get_logger().info(f"Published detections: {detection_message}")

            # Reset the current image
            self.current_image = None


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
