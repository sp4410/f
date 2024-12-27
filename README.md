# Further exploration: A Data-Driven Approach to Understanding Road Safety in NYC



Place `model.pt` and `model-SGD-0.01lr-50epoch.pt` into the `model&img` folder.

Visit the dataset website:

 [Chinese Traffic Sign Database](http://www.nlpr.ia.ac.cn/pal/trafficdata/recognition.html)



Download the compressed file `tsrd-train.zip` and move all the images and the file `label_test.txt` from the archive into the `test` folder.

|── train

​    |──000_0001.png

​    |──……

​    |──057_0003_j.png

​    |──label_train.txt

Download the compressed file `tsrd-test.zip` and move all the images and the file `label_train.txt` from the archive into the `train` folder.

|── test

​    |──000_0001_j.png

​    |──……

​    |──057_0002_j.png

​    |──label_test.txt





Download `bar pic.py`, `C&G&chart.py`, `CPU.py`, `dataload.py`, `diagram.py`, `distribution.py`, `main.py`, `README.md`, `resnet.py`, `test.py`, `train.py`, `trainfast.py`, `model.ckpt`, and `requirement.txt` to a folder on the desktop named `f`.

![image-20241226194940286](C:\Users\ThinkPad\Desktop\f\model&img\desktop.png)

## Repository Structure

├──f

​	├── model&img（model.pt & model-SGD-0.01lr-50epoch.pt）

​	├── test（all images and label_test.txt from the compressed file）

​	|── train（all images and label_train.txt from the compressed file）

​	├── bar pic.py: Peak hour chart

​	├── C&G&chart.py: Compare GPU and CPU performance

​	├── CPU.py: Generate heatmap & severity mapping

​	├── dataload.py        

​	├── diagram.py

​	├── GPU.py: GPU-generated severity mapping

​	├── main.py

​	├── model. ckpt

​	├── Motor_Vehicle_Collisions_-_Crashes_20241107.csv（Dataset from https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data）

​	├── README.md

​	├── requirement.txt

​	├── resnet.py

​	├── test.py

​	├── train.py（SGD optimizer output）

​	├── trainfast.py  

## Execute 

Open CMD and type:

`cd “……your_file……/Desktop/f"`

 run `pip install -r requirements.txt`.

then

Start training.

```
python main.py --train
```

(Install Torch and Triton on a Linux system) Perform optimized and accelerated training.

```
python main.py --train-faster
```

Perform test set accuracy statistics.

```
python main.py --test
```

Specify the target image for testing.

```
python main.py --test-img --image-path "./Test/000_0005_j.png"
```













CPU VS GPU

![image-20241225221209895](C:\Users\ThinkPad\Desktop\f\model&img\CPU vs GPU.png)

This result highlights the advantages of GPU parameters in data processing tasks, particularly in accelerating complex data analysis and machine learning workflows. By comparing the processing time, performance metrics, and visualization results between CPU and GPU, we can clearly observe the efficiency improvements and technical superiority brought by GPUs.



Use the GPU cluster to predict speed limit signs.

Input：python main.py --test-img --image-path "./Test/000_0005_j.png"

![image-20241227160538381](C:\Users\ThinkPad\Desktop\f\model&img\image-20241227160538381.png)





Use the GPU cluster to predict and focus on pedestrian signs.

Input：python main.py --test-img --image-path "./Test/035_1_0023_1_j.png"

![image-20241227160606338](C:\Users\ThinkPad\Desktop\f\model&img\image-20241227160606338.png)





Output Results of the SGD Optimizer

![image-20241225205448827](C:\Users\ThinkPad\Desktop\f\model&img\image-20241225205448827.png)

The SGD optimizer demonstrates efficient convergence and high accuracy in traffic sign classification tasks. After around 20 epochs, accuracy, recall, precision, and F1 scores approach 1.0, with the loss value stabilizing, making it suitable for high-precision requirements. Its optimization path is stable, offering strong generalization capabilities to adapt to dynamic traffic scenarios. Additionally, its low memory consumption improves GPU resource utilization, making it ideal for real-time traffic sign classification tasks based on the ResNet model.







