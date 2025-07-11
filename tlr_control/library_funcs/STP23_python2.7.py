# -*- coding: utf-8 -*-
import serial
import time
# 设置串口参数
# port = '/dev/wheeltec_STP23'  # 串口号,根据实际情况修改
port = '/dev/ttyACM0'
baudrate = 921600  # 波特率
def parse_data(data):
    # 检查起始符
    if data[0] != 0x54:
        print("Invalid data packet: Start byte not found.")
        return []
    
    measurements = []
    
    # 从第7个字节开始解析每个测量数据点
    for i in range(6, len(data)-5, 3):
        if i + 2 < len(data):  # 确保不会越界
            distance1 = data[i + 1]
            distance1=hex(distance1)       #将十进制整数转换为十六进制字符串表示
            interval1 = distance1[2:].upper()       #（去掉十六进制前面的0x）
            str1 = str(interval1)          #将 interval1 转为字符串

            distance2 = data[i]
            distance2 = hex(distance2)
            interval2 = distance2[2:].upper()
            str2= str(interval2)

            string1=str1+str2            #拼接高字节和低字节（用字符串来表示）

            distance =int(string1,16)    #将一个字符串 按照 16 进制的方式解析并转换为整数

            intensity = data[i + 2]  # 1字节信号强度

            measurements.append((distance, intensity))
    
    return measurements

def read_serial_data():
    with serial.Serial(port, baudrate, timeout=1) as ser:
        time.sleep(2)  # 等待设备准备
        while True:
            if ser.in_waiting > 0:
                raw_data = ser.read(ser.in_waiting)  # 读取所有可用数据
                data = list(bytearray(raw_data))  # 转换为列表以便处理
                measurements = parse_data(data)
                
                if measurements:
                    for idx, (distance, intensity) in enumerate(measurements):
                        print("测量点 {}: 距离值Distance = {} mm, 信号强度Intensity = {}".format(idx + 1, distance, intensity))
                    print("------------------------------")

if __name__ == "__main__":
    read_serial_data()


