## 运行环境
Python 3.4, TensorFlow 1.3, Keras 2.0.8, pycocotools以及requirements.txt中的其它包

训练好的权重于链接: https://pan.baidu.com/s/1C-z2Z_dy6CyXrhcJrBVxPw 提取码: 5ux1 下载，放在根目录下。

## 数据集
http://images.cocodataset.org/zips/test2015.zip
http://images.cocodataset.org/annotations/image_info_test2015.zip


## 测试训练好的模型
sample文件夹下有main.py文件，要测试现有图片，只需要将第55行：
```python
image = skimage.io.imread(os.path.join(IMAGE_DIR, r_image))
```
imread内改为要测试的图片的路径，用python运行该py文件，会输出一句对当前图片的描述。

