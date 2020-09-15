# Real-time-weld-defect-detection-using-Yolov3 (you only look once):
To prove the quality of products in pipeline manufacturing, manufacturers have to perform an X-ray inspection in compliance to ISO API 5L. Every weld has to be inspected and the result has to be archived. 
For that, Inspectors have to stay for more than seven hours scanning visually x-ray live video of the weld bead, this time-consuming task can bind a lot of manpower and cause a bottleneck in the production output. Moreover, it diminishes the efficiency of inspecting which works against the quality control duty.

Due to the machine utility,machine leaning in general and deep learning in particular based on artificial intelligence will ease the task and improve the control quality, so the automation of the task is right thing to solve this problem, and in order to automate radiographic-inspection task  in real-time, I had to choose the best object detection algorithm available. At present, Yolov3 architecture has shown magnificent performance in real-time detection which made it the right algorithm for this project.

In this work, I have prepared a training dataset and i have manually labeled 1300 defect image. Also, I have download yolov3 darknet weights from its website https://pjreddie.com/darknet/yolo/, which has already been trained on MSCOCO dataset and recognize 80 different classes with mAP (mean average precision)  measured at 0.5 IOU = 60.6 %  , also I had prepared all necessary files to be ready for training on Google Colab seeking the free and  powerful GPUs, I started training yolov3 model from the pre-trained weights, and at each number of epochs i have calculated the mAP (mean Average Precision) according to this code https://github.com/Cartucho/mAP and finally i have gotten these results:  
#Results of training yolov3 on custom x-ray dataset:
![training table](https://user-images.githubusercontent.com/47951668/92666441-2cc75800-f301-11ea-832b-dbdb7b417cd6.jpg)

According to these results, yolov3 weights at 3500 epochs are the best weights. and whenever I decrease the IOU rate the mAP increase. I decided to deploy these weights in my project.
I am using tensorflow and keras library. So, I had to convert weights from darknet format (.weights) to pyh5 format (.h5) to be suitable for deployment in keras, i did converted it with the use of  this code:  https://github.com/qqwweee/keras-yolo3/blob/master/convert.py

To customize the application I had to simulate the operator work, and I end up with three main operations:

- a)	Grab the image from the x-ray detector software “YXLON Image 3500” (first, the operator look visually to the video).
- b)	Analyzing the frame image using that converted weights and get results. (the operator inspect the frame in his mind) 
- c)	Give orders according to these results. (the operator take a decision according to result, whether to stop the operation and mention the defect location on the tube or just continue when the defect is acceptable)

The objective of this project is to build an application that can imitate exactly these operations. As It is well known that detecting objects in real time requires powerful GPUs. Therefore, I decided to ascribe the detection task to a separated PC. For that, I had to split the application into two parts: Server and detection, using two separated PCs connected over a Local network.

The server part has to be installed on the main PC and do these tasks:
- Grab the current image from the YXLON software (Image3500)
- Convert that image to bytes and send it to the detection PC
- Receive orders from the detection PC according the the detection results.
- Integrate, archive and print the  image when necessary. 
- Commanding the micro-controller board (arduino uno in this case) which has to do: 
1.	The control of chariot's movement (stop, run and pause)
2.	The command of paint injectors to localize the defect on the pipe. 
3.	Receive security conditions from optic sensors and secure the operation according to these conditions.

The detection application part has to be installed on the detection PC which contains GPUs and do these tasks:
- 	Receive data from Server and convert it to image
- 	Feed the yolov3 model with the received image and store the results.
- 	Depending on these results, choose the correct order and send it to Server.
- 	Measuring the detection speed (the FPS rate) and depending on that, computing the inspection speed limit.

On my laptops, I am using NVIDIA GE Force 840M.  I optimized GPU consumption and synchronize all these tasks using multi threading techniques, despite that I recorded only 1.5 FPS which is still slow (maximum 30cm/s).
On the site, I will use NVIDIA GTX 1080 Ti 11g or RTX 2080 Ti 11g. So I will see what speed the application can reach.


# Requirements:
In this project I have used the following libraries and tools:
1. Python 3.7.7
2. Keras 2.2.4
3. H5py 2.10.0
4. Tensorflow-gpu 1.13.1 & cpu(if there is no gpu) 
5. Cuda toolkit (depending on the operation system)
6. Cudnn (compatible to the OS and the Cuda version)
7. Opencv 4.2.0.34
8. Easygui 0.98.1
9. Pillow 7.1.1
10. Numpy1.19.1
11. Matplotlib 3.2.1
12. Mss 6.0.0
13. PyAutoGui 0.9.50
14. Pyserial 3.4
15. pyinstaller 3.6
for direct use, you can download the executable of both parts of the application and use them just by double click:
- Server exe for windows 64 and 32 bit:
- Detection part exe for windows 64 bit only: gpu ... cpu ...
- The yolov3 pre-trained model in h5 format:
# Demonstrative video:
This video shows a brief background with the source code and simulates the use of the application: https://youtu.be/yJJ3wPdUuPM

# The user manual of the app:
Once the exe files have been downloaded. on Server PC, double click on Server.exe, a window (fig: 1) will appear:
1.	Port used for the first thread 
2.	Port used for the second thread
3.	The arduino COM port
4.	Integration time (in seconds)
5.	The pipe name (for each tube the operator should fill up this field)
6.	The current number of integration (per tube)
7.	Click this to start listening from the detection PC
8.	Empty signal field to show the current operation on the app 
9.	Click on this button to close the app
10.	Start the chariot manually 
11.	Pause the chariot manually
12.	Move the chariot back manually
13.	Integrate the current image manually
14.	Check it to choose to save the image after integration.
15.	 Check it to choose to print the image after integration.
16.	The path of the global folder in which the app saves images

![fig 1](https://user-images.githubusercontent.com/47951668/92667311-803aa580-f303-11ea-8886-08027fc8e500.jpg)

Once you fill up all fields, click on connect button to start listening from the detection PC (fig-2).

![fig2](https://user-images.githubusercontent.com/47951668/92667374-ac562680-f303-11ea-8a82-6fa23d301101.jpg)

# On the detection PC (second part of the application):
Double click on detection executable a window (fig:3) will appear (loading tensorflow and model weights may take a while):
1. The IP address of the server PC on the local network
2. The port used for first thread
3. The port used for second thread
4. Defect dimension threshold (over it, the integration process will start)
5. Set the number of defect threshold (over it, the integration process will start)
6. Set the confidence threshold (under it, defects will not be considered as correct defect)
7. Set the IOU  (intersection over union) rate threshold. (threshold to choose the right anchor to predict boxes, it is inversely proportional with accuracy) 
8. The admissible  inspection speed (on meter per second m/s) 
9. Click to start data exchange with the server
10. Click to quit and close the app

Make sure ports and IP adresse are identical to server ones then click on start button to connect to the Server.  .

![fig3](https://user-images.githubusercontent.com/47951668/92667477-ef17fe80-f303-11ea-876d-d3d6239b229f.jpg)

# On Server PC: 

Digital x-ray detector software (YXLON image 3500) is an image editor, it show the current x-ray image(Fig:4), The application is configured to grab, convert and send the selected red square area (1000X1000 pixels) in real-time to detection PC for analyzing it using the pre-trained model YOLOV3. 

![fig4](https://user-images.githubusercontent.com/47951668/92667480-f3441c00-f303-11ea-945e-fc72538a09f5.jpg)
#
-	Once the detection PC is started,  the color of signal field in the server app turned green and indicate that the first thread is started and the detection pc is connected 
(fig: 5).
![fig5](https://user-images.githubusercontent.com/47951668/92667537-1a9ae900-f304-11ea-9ce3-0a9b35e9f002.jpg)

# On the detection PC:
After getting connected, automatically a window will appear showing each defect inside a blue bounding box with name and confidence rate on the current grabbed image after being analyzed (fig: 6)

![fig6](https://user-images.githubusercontent.com/47951668/92667593-44541000-f304-11ea-9075-0aa6165210db.jpg)

On the left side in figure 6, all information about defects are given and been updated each frame, speed limit=0.03 m/s which is really slow (3cm/s) due to the small FPS = 0.1 thus, here, I used tensorflow-CPU to show the minimum computing power required to run this application.

# On the server PC:
As it is shown there are 3 defects has been detected, so, an order has been sent to server to stop the chariot and start the integration process.
The signal field Label and background have been changed, it indicates that integration process has been started (fig: 7):

![fig7](https://user-images.githubusercontent.com/47951668/92667667-706f9100-f304-11ea-82bf-1b5a7a85f1c9.jpg)

After finishing image integration on YXLON software,if save checkbox is checked, the application is configured to save the image according to 1,2,3 in the fig: 7 respectively. 
According to the given path (1) the app will create a folder named as PIPE NAME (2) and inside that folder, the integrated image will be archived under the name of the pipe (2) concatenated with the corresponded number of integration (3) as follows:

![noname](https://user-images.githubusercontent.com/47951668/92667686-7ebdad00-f304-11ea-884e-f644311f3c3e.jpg)

also, if checkbox “print” is checked, the integrated image will be printed. finally, the server will start the chariot again to start another cycle and the loop will continue until the end of the tube.

# Manual command and inspection:
When the operator wants to detect defects visually, the server application can be used for manual inspection mode as follows:

- Click on START button to start the chariot movement, the START button background will be changed to green and the label will be changed to CH_STARTED (01 in fig: 8).
- Click on PAUSE button to stop the chariot instantly, the button background will change to orange and the label will be RESUME means if you want to continue just click again (02 in fig: 8)
- Click on BACK button to pull the chariot back, the chariot will move backward and the button background color will change to red and the label will change to STOP! Means if you want to stop just click again (03 in fig: 8)
- Click on INTEGRATE button to start integration process manually of the current image the x-ray software, the button background color will change to yellow and after finishing integration it return white indicting that integration has finished (04 in fig: 8)

![fig8](https://user-images.githubusercontent.com/47951668/92667727-a57be380-f304-11ea-8e0d-f87052d9c03a.jpg)
# Contact:
b.yassine.89@gmail.com
