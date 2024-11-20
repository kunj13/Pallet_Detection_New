import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
from torchvision import io, transforms as T
from PIL import Image as PILImage
import numpy as np
import os
from pathlib import Path
from semseg.models import SegFormer
from semseg.datasets import ADE20K  # Adjust based on your dataset

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')

        # Initialize ROS2 publishers and subscribers
        self.image_subscriber = self.create_subscription(
            Image, '/processed/annotated_image', self.image_callback, 10)
        self.segmented_image_publisher = self.create_publisher(Image, '/processed/segmented_image', 10)
        
        # Initialize bridge for ROS-OpenCV conversion
        self.bridge = CvBridge()

        # Load SegFormer model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SegFormer(backbone='MiT-B3', num_classes=150).to(self.device)
        
        # Load model weights
        weights_path = '/home/kunj/pallet_detect_ws/src/my_package/config/segformer.b3.ade.pth'  # Update with actual path
        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.get_logger().info('Loaded Model')
        except Exception as e:
            self.get_logger().error(f"Failed to load model weights: {e}")
        self.model.eval()

        # Define transformations
        self.preprocess = T.Compose([
            T.CenterCrop((512, 512)),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.palette = ADE20K.PALETTE  # Color palette for segmentation

    def image_callback(self, msg):
        # Convert ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        
        # Convert OpenCV image to PIL and preprocess
        pil_image = PILImage.fromarray(cv_image)
        image = T.ToTensor()(pil_image).to(self.device)  # Convert to tensor and move to device
        image = self.preprocess(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            seg = self.model(image)
        seg = seg.softmax(1).argmax(1).squeeze().cpu().numpy()

        # Map segmentation output to color palette
        seg_colored = self.palette[seg].astype(np.uint8)

        # Convert segmented output back to ROS Image and publish
        segmented_msg = self.bridge.cv2_to_imgmsg(seg_colored, encoding='rgb8')
        self.segmented_image_publisher.publish(segmented_msg)
        self.get_logger().info("Published segmented image.")

def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()