import socket
import sys
import os
import numpy as np
import cv2
import time

# 服务器端IP和端口
windows_host = "192.168.137.1"  # 你的Windows系统IP地址
windows_port = 20727  # 选择一个可用的端口号


def read_image(filename):
    try:
        # 以二进制方式打开图片文件
        with open(filename, "rb") as f:
            image_data = f.read()
            return image_data
    except Exception as e:
        raise Exception("无法读取图片文件: " + str(e))


def delete_photo(img_path):
    try:
        # 检查照片是否存在
        if os.path.exists(img_path):
            # 删除照片
            os.remove(img_path)
            print("照片已成功删除：", img_path)
        else:
            print("照片不存在：", img_path)
    except OSError as e:
        print("删除照片时出错：", e)


def main():
    # 创建TCP连接
    image_path = "pi_send_img.jpg"
    # 读取图片数据
    try:
        image_data = read_image(image_path)
    except Exception as e:
        print("发生错误：", e)
        return

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
        try:
            expected_image_size = len(image_data)
            client_socket.sendto(str(expected_image_size).encode("utf-8"), (windows_host, windows_port))
            # 等待接收确认消息
            ack_message, _ = client_socket.recvfrom(1024)
            print("接收到确认消息：", ack_message.decode("utf-8"))

            # 发送图像数据
            buffer_size = 4096  # 每次发送的缓冲区大小
            offset = 0  # 已发送的数据偏移量
            while offset < expected_image_size:
                end_offset = min(offset + buffer_size, expected_image_size)
                client_socket.sendto(image_data[offset:end_offset], (windows_host, windows_port))
                offset = end_offset

            print("图片数据发送成功！")

            feedback, _ = client_socket.recvfrom(1024)
            print("收到反馈：", feedback.decode("utf-8"))
            delete_photo(image_path)
            return feedback.decode("utf-8")
        except Exception as e:
            print("发生错误：", e)


def pi_send_img():
    feedback = main()
    return feedback

if __name__ == "__main__":
    main()