from setuptools import setup
import os

package_name = 'my_package'

def package_files(directory):
    # Recursively gather all files in the directory
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths
yolov5_files = package_files(os.path.join(package_name, 'yolov5'))

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (package_name, ['my_package/GroundingDINO_SwinT_OGC.py']),
        (package_name, ['my_package/initial_data_processing_node.py']),
        (package_name, ['my_package/segment_node.py']),
        (package_name, ['my_package/yolo_inference_node.py']),
        # ('share/my_package/config', [
        # 'config/groundingdino_swint_ogc.pth',
        # ]),
        # (os.path.join('share', package_name, 'config'), [
        # 'config/segformer.b3.ade.pth',  # Add this line
        # ]),
        ('lib/'+ package_name+ '/yolov5', [
            os.path.join(package_name, 'yolov5/export.py'),
        ]),
        ('lib/' + package_name + '/yolov5/utils', [
            os.path.join(package_name, 'yolov5/utils/plots.py'),
            os.path.join(package_name, 'yolov5/utils/autoanchor.py'),
            os.path.join(package_name, 'yolov5/utils/augmentations.py'),
            os.path.join(package_name, 'yolov5/utils/dataloaders.py'),
            os.path.join(package_name, 'yolov5/utils/__init__.py'),
            os.path.join(package_name, 'yolov5/utils/metrics.py'),
            os.path.join(package_name, 'yolov5/utils/downloads.py'),
            os.path.join(package_name, 'yolov5/utils/general.py'),
            os.path.join(package_name, 'yolov5/utils/torch_utils.py'),
        ]),
        ('lib/' + package_name + '/yolov5/models', [
            os.path.join(package_name, 'yolov5/models/experimental.py'),
            os.path.join(package_name, 'yolov5/models/common.py'),
            os.path.join(package_name, 'yolov5/models/yolo.py'),
        ]), 
    ],
    install_requires=['setuptools'],  # This remains for required Python packages
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='A ROS2 Python package',
    license='License Declaration',
    entry_points={
        'console_scripts': [
            # 'node_name = my_package.node_script:main',
            'initial_data_processing = my_package.initial_data_processing_node:main',
            'segment = my_package.segment_node:main',
            'yolo_inference = my_package.yolo_inference_node:main',
        ],
    },
)
