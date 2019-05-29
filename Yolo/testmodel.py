import os,time
aaa=os.popen("ls")
path="/home/zyh/文档/officialyolo/darknet-master/valdataset/"


files=os.listdir(path)
for file in files:
	aaa=os.popen("./darknet detector test cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny_10000.weights "+path+file)
	time.sleep(10)

