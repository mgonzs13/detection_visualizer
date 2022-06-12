# Copyright 2019 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import random

import cv2
import cv_bridge
import message_filters
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


class DetectionVisualizerNode(Node):

    def __init__(self):
        super().__init__('detection_visualizer')

        self._bridge = cv_bridge.CvBridge()

        # load class names for color visualization
        self.declare_parameter("class_names", "")
        class_names_files = self.get_parameter("class_names").get_parameter_value().string_value
        
        self._class_to_color = {}

        try:
            f = open(class_names_files, "r")
        
            for line in f:
                r = random.randint(0,255)
                g = random.randint(0,255)
                b = random.randint(0,255)
                self._class_to_color[line.strip()] = (r, g, b)
        except FileNotFoundError:
            self._class_to_color = (0, 255, 0)

        output_image_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=1)

        self._image_pub = self.create_publisher(Image, 'dbg_images', output_image_qos)

        self._image_sub = message_filters.Subscriber(self, Image, 'images')
        self._detections_sub = message_filters.Subscriber(self, Detection2DArray, 'detections')

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self._image_sub, self._detections_sub), 10, 0.5)
        self._synchronizer.registerCallback(self.on_detections)

    def on_detections(self, image_msg: Image, detections_msg: Detection2DArray):
        cv_image = self._bridge.imgmsg_to_cv2(image_msg)

        # Draw boxes on image
        for detection in detections_msg.detections:
            max_class = None
            max_score = 0.0
            for hypothesis in detection.results:
                if hypothesis.score > max_score:
                    max_score = hypothesis.score
                    max_class = hypothesis.id
            if max_class is None:
                print("Failed to find class with highest score", file=sys.stderr)
                return

            cx = detection.bbox.center.x
            cy = detection.bbox.center.y
            sx = detection.bbox.size_x
            sy = detection.bbox.size_y

            color = None
            if isinstance(self._class_to_color, dict):
                color = self._class_to_color[max_class]
            else:
                color = self._class_to_color

            min_pt = (round(cx - sx / 2.0), round(cy - sy / 2.0))
            max_pt = (round(cx + sx / 2.0), round(cy + sy / 2.0))
            thickness = 2
            cv2.rectangle(cv_image, min_pt, max_pt, color, thickness)

            label = '{} {:.3f}'.format(max_class, max_score)
            pos = (min_pt[0], max_pt[1])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cv_image, label, pos, font, 0.75, color, 1, cv2.LINE_AA)
            
        detection_image_msg = self._bridge.cv2_to_imgmsg(cv_image, encoding=image_msg.encoding)
        detection_image_msg.header = image_msg.header

        self._image_pub.publish(detection_image_msg)


def main():
    rclpy.init()
    rclpy.spin(DetectionVisualizerNode())
    rclpy.shutdown()
