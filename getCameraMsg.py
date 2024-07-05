# import airsim # pip install airsim
# import numpy as np
# import os
# import cv2




# def process_rgb_msg2numpyarray(rgb_response):

#     # Check if the RGB response is valid
#     if rgb_response.width > 0 and rgb_response.height > 0:
#         # Get numpy array from the RGB response
#         img_rgb1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)

#         # Ensure the array size matches the expected dimensions
#         if img_rgb1d.size == rgb_response.width * rgb_response.height * 3:
#             # Reshape array to 3 channel image array H X W X 3
#             img_rgb = img_rgb1d.reshape(rgb_response.height, rgb_response.width, 3)

#             # Original image is flipped vertically
#             # img_rgb = np.flipud(img_rgb)
#             # # Display the image in real-time
#             cv2.imshow("RGB Image", img_rgb)
#             cv2.waitKey(0)  # Wait for 1 ms to display the image
#             # filename = "test_img"
#             # # Write to png 
#             # airsim.write_png(os.path.normpath(filename + '.png'), img_rgb)

#         else:
#             print("The size of the received RGB image data does not match the expected dimensions.")
#     else:
#         print("Invalid RGB response received.")
#     return img_rgb


# def process_depth_msg2numpyarray(depth_response):

#     # 处理深度图像
#     if depth_response.width > 0 and depth_response.height > 0:
#         # 将深度图像数据转换为 numpy 数组
#         img_depth1d = np.array(depth_response.image_data_float, dtype=np.float32)

#         if img_depth1d.size == depth_response.width * depth_response.height:
#             img_depth = img_depth1d.reshape(depth_response.height, depth_response.width)

#             # 将深度图像转换为 8 位单通道图像以便显示
#             # img_depth_visual = cv2.normalize(img_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#             # 实时显示深度图像
#             cv2.imshow("Depth Image", img_depth)
#             cv2.waitKey(0)  # Wait for 1 ms to display the image

#         else:
#             print("接收到的深度图像数据大小与预期维度不匹配。")
#     else:
#         print("接收到的深度响应无效。")



# if __name__ == "__main__":

#     # Connect to the AirSim simulator
#     client = airsim.MultirotorClient()
#     client.confirmConnection()

#     # Request images from the simulator
#     responses = client.simGetImages([
#         airsim.ImageRequest(0, airsim.ImageType.Scene, False, False),
#         airsim.ImageRequest('front_center', airsim.ImageType.DepthPlanar, True, False)
#         ])


#     rgb_response = responses[0]
#     depth_response = responses[1]

#     process_rgb_msg2numpyarray(rgb_response)
#     process_depth_msg2numpyarray(depth_response)

import airsim  # pip install airsim
import numpy as np
import cv2
import threading
import time

# 全局变量用于存储图像
rgb_image = None
depth_image = None

def process_rgb_msg2numpyarray(rgb_response):
    global rgb_image
    # Check if the RGB response is valid
    if rgb_response.width > 0 and rgb_response.height > 0:
        # Get numpy array from the RGB response
        img_rgb1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)

        # Ensure the array size matches the expected dimensions
        if img_rgb1d.size == rgb_response.width * rgb_response.height * 3:
            # Reshape array to 3 channel image array H X W X 3
            rgb_image = img_rgb1d.reshape(rgb_response.height, rgb_response.width, 3)
        else:
            print("The size of the received RGB image data does not match the expected dimensions.")
    else:
        print("Invalid RGB response received.")



def process_depth_msg2numpyarray(depth_response):
    global depth_image
    # 处理深度图像
    if depth_response.width > 0 and depth_response.height > 0:
        # 将深度图像数据转换为 numpy 数组
        img_depth1d = np.array(depth_response.image_data_float, dtype=np.float32)

        if img_depth1d.size == depth_response.width * depth_response.height:
            img_depth = img_depth1d.reshape(depth_response.height, depth_response.width)
            depth_image=img_depth
            # 将深度图像转换为 8 位单通道图像以便显示
            # depth_image = cv2.normalize(img_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            print("接收到的深度图像数据大小与预期维度不匹配。")
    else:
        print("接收到的深度响应无效。")


def fetch_and_process_images(client):
    while True:
        # Request images from the simulator
        responses = client.simGetImages([
            airsim.ImageRequest(0, airsim.ImageType.Scene, False, False),
            airsim.ImageRequest(0, airsim.ImageType.DepthVis, True, False)
        ])

        rgb_response = responses[0]
        depth_response = responses[1]

        # print("rgb_response :", rgb_response.width , " height:", rgb_response.height)
        # print("depth_response :", depth_response.width , " height:", depth_response.height)

        process_rgb_msg2numpyarray(rgb_response)
        process_depth_msg2numpyarray(depth_response)

        # 以10Hz的频率读取和显示图像
        time.sleep(0.05)



def main():
    global rgb_image, depth_image

    # Connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # 创建一个线程来定时读取和处理图像
    image_thread = threading.Thread(target=fetch_and_process_images, args=(client,))
    image_thread.start()

    # 主线程用于显示图像并处理键盘事件
    while True:
        if rgb_image is not None:
            cv2.imshow("RGB Image", rgb_image)
        if depth_image is not None:
            cv2.imshow("Depth Image", depth_image)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()

