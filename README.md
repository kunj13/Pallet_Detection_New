# 🚀 Pallet Detection and Segmentation with ROS2 and Custom AI Models

Welcome to the **Pallet Detection and Segmentation** project! This repository leverages **ROS2** and **deep learning models** for real-time object detection and segmentation of pallets in manufacturing or warehousing environments. Designed to support mobile robotics applications, this project enables efficient pallet tracking and scene understanding on-the-go!

## 🌟 Project Overview
In dynamic warehouse environments, accurate detection and segmentation of pallets are critical for automated handling and inventory management. This project combines object detection (using YOLO) and semantic segmentation (using SegFormer) into a cohesive ROS2 pipeline. Whether you’re building an inventory bot or managing a smart warehouse, this system’s modular design ensures adaptability and easy integration.

### 🚧 System Requirements
- **Operating System**: Ubuntu 20.04 recommended
- **Hardware**: NVIDIA GPU with CUDA support for optimal inference
- **Software**: ROS2 Galactic, Python 3.8+

### 🔧 Dependencies and Installations
To get started, clone the repository and install all dependencies!

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/kunj13/Pallet_Detection_New.git
    cd Pallet_Detection_New
    ```

2. **Install ROS2 Galactic**  
   *Follow [these instructions](https://docs.ros.org/en/galactic/Installation.html) for ROS2 Galactic installation.*

3. **Install Required Python Libraries**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Include packages like `torch`, `torchvision`, `cv_bridge`, `sensor_msgs`, and `opencv-python`)*

4. **Set Up ROS2 Workspace**:
    ```bash
    mkdir -p ~/pallet_ws/src
    cd ~/pallet_ws/src
    git clone https://github.com/kunj13/Pallet_Detection_New.git
    cd ~/pallet_ws
    colcon build
    ```

### 💾 Model Checkpoints
For accurate pallet detection and segmentation, download the pretrained model weights from Google Drive:

Download the config folder for the following - [Download here from Google Drive](https://drive.google.com/drive/folders/1FcEHy29rdMRAS52c0Sp4se8oW6ygwvz5?usp=sharing)
- **Custom YOLO Detection Weights** - Best.pt
- **Custom SegFormer Segmentation Weights** - segformer.b3.ade.pth
- **Custom Grounding Dino weights** - groundingdino_swint_ogc.pth

> **After downloading**: Place these weights in the `src/my_package/config/` directory of your workspace.

---

## 🛠️ Workflow Overview
Here’s a step-by-step breakdown of the nodes and data flow:

1. **Data Processing Node** (`initial_data_processing`):
   - Subscribes to camera feed and preprocesses the data for detection.
2. **YOLO Detection Node** (`yolo_inference`):
   - Performs real-time object detection, identifying pallets with bounding boxes.
3. **Segmentation Node** (`segment`):
   - Applies semantic segmentation to detect pallet and ground areas.
4. **Result Publisher**:
   - Publishes final annotated images and segmentation maps to specified topics.

---

### 🚀 Running the Nodes
Use the following commands to start each node. Ensure your ROS2 workspace is sourced!

```bash
# Start the Initial Data Processing Node
ros2 run my_package initial_data_processing

# Start the YOLO Detection Node
ros2 run my_package yolo_inference

# Start the Segmentation Node
ros2 run my_package segment
```

---

## ✨ Example Usage
To process sample data, simply run the nodes as shown above. Publish sample images to the specified topics, and watch the system annotate and segment pallets in real time!

---

## 🛠️ Troubleshooting
- **Model Loading Errors**: Double-check that the model weights are correctly downloaded and placed in the `config` directory.
- **Topic Compatibility**: Ensure your image topics are compatible with the nodes’ required format (`rgb8` for example).
- **Performance on CPU**: This project performs best on a GPU. Running on a CPU may significantly slow down inference.

---

## 🎉 Acknowledgments
This project leverages open-source contributions from the ROS and AI communities. Special thanks to [SegFormer](https://github.com/NVIDIA/semantic-segmentation) (semantic segmentation model) and [YOLOv5](https://github.com/ultralytics/yolov5) (object detection model) for their models and insights.

---

## 📊 Test Results
In the `src/my_package/test_results` folder, you'll find an example of the pipeline’s output on a sample image. This includes:
- The **original image**
- The **YOLO-processed image** with bounding boxes
- The **segmented image** from SegFormer

These samples demonstrate the model’s effectiveness in detecting and segmenting pallets in real warehouse environments.


