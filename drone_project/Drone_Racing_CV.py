import rclpy
import time
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt
import sys
from tello_msgs.srv import TelloAction
from djitellopy import Tello
import cv_bridge

# tello = Tello()

class Tello(Node):
    # global k
    def __init__(self):
        super().__init__('tello')
        qos_policy = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.bridge = cv_bridge.CvBridge()
        # To Send the takeoff and land commands
        self.cli = self.create_client(TelloAction, '/drone1/tello_action')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = TelloAction.Request()
        self.req.cmd = 'takeoff'
        self.future = self.cli.call_async(self.req)
        time.sleep(3)
        self.publisher_ = self.create_publisher(Twist, '/drone1/cmd_vel', 10)
        self.subscription = self.create_subscription(
            Image,
            '/drone1/image_raw',
            self.camera_callback,
            qos_profile=qos_policy 
        )
    
    def camera_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            imageFrame=cv2.resize(image, (780, 540),
                interpolation = cv2.INTER_LINEAR)
            imageFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2RGB)
        # # imageFrame=image
            # Display the image
            cv2.imshow("Image.jpg", imageFrame)
            cv2.waitKey(1)  # Adjust delay as needed
        except cv_bridge.CvBridgeError as e:
            print("Failed to convert ROS Image message:", e)


        # hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_RGB2HSV)
        # green_lower = np.array([25, 52, 72], np.uint8)
        # green_upper = np.array([102, 255, 255], np.uint8)
        # green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

        # red_lower = np.array([111, 87, 136], np.uint8)
        # red_upper = np.array([180, 255, 255], np.uint8)
        # red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

        # blue_lower = np.array([94, 80, 2], np.uint8)
        # blue_upper = np.array([120, 255, 255], np.uint8)
        # blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

        # kernel = np.ones((5, 5), "uint8")

        # green_mask = cv2.dilate(green_mask, kernel)
        # res_green = cv2.bitwise_and(imageFrame, imageFrame,
        #                             mask = green_mask)
        # # For red color
        # red_mask = cv2.dilate(red_mask, kernel)
        # res_red = cv2.bitwise_and(imageFrame, imageFrame, 
        #                         mask = red_mask)

        # # For blue color
        # blue_mask = cv2.dilate(blue_mask, kernel)
        # res_blue = cv2.bitwise_and(imageFrame, imageFrame,
        #                         mask = blue_mask) 

        #  # Creating contour to track red color
        # contours_red, hierarchy = cv2.findContours(red_mask,
        #                                    cv2.RETR_TREE,
        #                                    cv2.CHAIN_APPROX_SIMPLE)
      
        # for pic, contour in enumerate(contours_red):
        #     area = cv2.contourArea(contour)
        #     if(area > 300):
        #         x, y, w, h = cv2.boundingRect(contour)
        #         imageFrame = cv2.rectangle(imageFrame, (x, y), 
        #                                 (x + w, y + h), 
        #                                 (0, 0, 255), 2)

        #  # Creating contour to track green color
        # contours_green, hierarchy = cv2.findContours(green_mask,
        #                                    cv2.RETR_TREE,
        #                                    cv2.CHAIN_APPROX_SIMPLE)
      
        # for pic, contour in enumerate(contours_green):
        #     area = cv2.contourArea(contour)
        #     if(area > 300):
        #         x, y, w, h = cv2.boundingRect(contour)
        #         imageFrame = cv2.rectangle(imageFrame, (x, y), 
        #                                 (x + w, y + h),
        #                                 (0, 255, 0), 2)

        # # Creating contour to track blue color
        # contours_blue, hierarchy = cv2.findContours(blue_mask,
        #                                     cv2.RETR_TREE,
        #                                     cv2.CHAIN_APPROX_SIMPLE)
        # for pic, contour in enumerate(contours_blue):
        #     area = cv2.contourArea(contour)
        #     if(area > 300):
        #         x, y, w, h = cv2.boundingRect(contour)
        #         imageFrame = cv2.rectangle(imageFrame, (x, y),
        #                                 (x + w, y + h),
        #                                 (255, 0, 0), 2)

        # print(len(contours_green), "contours")
        # shape = self.shape_detection(res_green)
        # print(shape,"shape")
        # if (len(contours_green) and (shape==1 or shape ==2)):

        #     print("detected")
        #         # area = cv2.contourArea(contour)
        #         # cntrs = list(contours)
        #     # list for storing names of shapes
        #         # cntours = []
        #         # for pos,contour in enumerate(contours_green):
        #         #     area = cv2.contourArea(contour)
        #         #     print(area)
        #         #     if area <30000 and area >20000:
        #         #         cntours.append(contour)           
        #         #     # using drawContours() function
        #         # print(len(cntours))
        #     M = cv2.moments(contours_green[1])
        #     if M['m00'] != 0.0:
        #         pos_x = int(M['m10']/M['m00'])
        #         pos_y = int(M['m01']/M['m00'])
        #     cv2.circle(imageFrame, (pos_x, pos_y),3,(0, 0, 255),-1)
        #     if pos_x < 399:
        #         print("I am in pos_x < 350" )
        #         cmd_publish = Twist()
        #         # cmd_publish.linear.y= -0.04
        #         # cmd_publish.linear.x = 0.02
        #         cmd_publish.angular.z = 0.02
        #         self.publisher_.publish(cmd_publish)
        #         time.sleep(0.1)
        #         # cmd_publish.linear.y= 0.0
        #         cmd_publish.angular.z = 0.0
        #         # cmd_publish.linear.x = 0.0
        #         self.publisher_.publish(cmd_publish) 
        #     if pos_x> 401:
        #         print("I am in pos_x > 450" )
        #         cmd_publish = Twist()
        #         # cmd_publish.linear.y= +0.04
        #         # cmd_publish.linear.x = 0.02
        #         cmd_publish.angular.z = -0.02
        #         self.publisher_.publish(cmd_publish)
        #         time.sleep(0.1)
        #         cmd_publish.linear.y= 0.04
        #         cmd_publish.linear.x = 0.0
        #         cmd_publish.angular.z = 0.0
        #         self.publisher_.publish(cmd_publish) 
        #     if (pos_x>399 and pos_x<401):
        #         # global k
        #         print("I am in the last")
        #         # k = 0
        #         cmd_publish = Twist()
        #         # cmd_publish.angular.z = 0.
        #         cmd_publish.linear.x = 0.5
                
        #         self.publisher_.publish(cmd_publish)
        #         time.sleep(2)
        #         cmd_publish.linear.x = 0.0
        #         self.publisher_.publish(cmd_publish)
                
        # else:
        #     # global k
        #     # k+=1
        #     print("not detected so rotating for detection")
        #     cmd_publish = Twist()
        #     cmd_publish.angular.z = 0.15
        #     self.publisher_.publish(cmd_publish)
        #     time.sleep(0.1)
        #     cmd_publish.angular.z = 0.0
        #     self.publisher_.publish(cmd_publish)
        
        
        
        # # cntrs = list(contours)
        # # list for storing names of shapes
        # cntours = []
        # for pos,contour in enumerate(contours):
        #     area = cv2.contourArea(contour)
        #     print(area)
        #     if area > 5000:
        #         cntours.append(contour)           
        #     # using drawContours() function
        # print(len(cntours))
       
        # cv2.drawContours(res_green, [cntours[1]], 0, (0, 0, 255), 5)
        
        #     # finding center point of shape
        # M = cv2.moments(contour)
        # if M['m00'] != 0.0:
        #     x = int(M['m10']/M['m00'])
        #     y = int(M['m01']/M['m00'])
        # cv2.circle(res_green, (x, y),3,(0, 0, 255),-1)
        # cv2.imshow("image", res_green)
        # cv2.waitKey(1)

    def shape_detection(self, image):
        h, w = image.shape[0:2]
        neww = 600
        newh = int(neww*(h/w))
        image = cv2.resize(image, (neww, newh))

        cv2.imshow("image", image)
        cv2.waitKey(-1)
        
        # converting image into grayscale image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # setting threshold of gray image
        _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # using a findContours() function
        contours, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        i = 0
        
        # list for storing names of shapes
        for contour in contours:
        
            # here we are ignoring first counter because 
            # findcontour function detects whole image as shape
            if i == 0:
                i = 1
                continue
        
            # cv2.approxPloyDP() function to approximate the shape
            approx = cv2.approxPolyDP(
                contour, 0.01 * cv2.arcLength(contour, True), True)
            
            # using drawContours() function
            cv2.drawContours(image, [contour], 0, (0, 0, 255), 5)
        
            # finding center point of shape
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
        
            # putting shape name at center of each shape
            if len(approx) == 4:
                cv2.putText(image, 'Quadrilateral', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                return 1
            else:
                cv2.putText(image, 'circle', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                return 2
            

    def imgmsg_to_cv2(self, img_msg):

        n_channels = len(img_msg.data) // (img_msg.height * img_msg.width)
        dtype = np.uint8

        img_buf = np.asarray(img_msg.data, dtype=dtype) if isinstance(img_msg.data, list) else img_msg.data

        if n_channels == 1:
            cv2_img = np.ndarray(shape=(img_msg.height, img_msg.width), dtype=dtype, buffer=img_buf)
        else:
            cv2_img = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels), dtype=dtype, buffer=img_buf)

        # If the byte order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            cv2_img = cv2_img.byteswap().newbyteorder()

        return cv2_img
    
    

def main(args=None):
    global k 
    k = 0
    rclpy.init(args=args)
    tello = Tello()
    rclpy.spin(tello) 
    tello.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
