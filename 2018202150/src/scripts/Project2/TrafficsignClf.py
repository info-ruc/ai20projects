import os
from torchvision import transforms,models
import torch.nn as nn
import torch
import PIL

# 用来预测的分类器
class TrafficsignClf():
	# 构造函数
	# 参数:
	#		model_path: 模型所在**绝对**路径
	def __init__(self, model_path):
		# 有 8 类
		self.classes = ['speed_limit_30', 'speed_limit_40', 
			'go_straight', 'turn_left', 'turn_right', 
			'turn_around', 'slow', 'stop']
		self.model = models.resnet18()
		num_ftrs = self.model.fc.in_features
		self.model.fc = nn.Linear(num_ftrs, 9)
		# 加载模型
		self.model.load_state_dict(torch.load(model_path))
		self.model.eval()
		
	# 预测
	# 参数:
	#		img_path: 所需预测的图片所在的**绝对**路径
	def predict(self, img_path):
		img = PIL.Image.open(img_path).convert('RGB')
		test_transform = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
		img = test_transform(img)
		batch_t = torch.unsqueeze(img, 0)
		output = self.model(batch_t)
		_, predict = torch.max(output, 1)
		# 返回类别标签和类别名
		# 如果不把握，返回没有交通标志
		if int(_) < 3:
			return 8, "N/A"
		return int(predict), self.classes[int(predict)]

if __name__ == '__main__':
	clf = TrafficsignClf(os.getcwd() + '/model_resave.pth')
	# 从摄像头获取画面
	cap = cv2.VideoCapture(1)
	ret, frame = cap.read()
	if ret:
		# 暂时存储摄像头拍摄到的帧至此
		name = os.getcwd() + '/frame.jpg'
		cv2.imwrite(name, frame)
		_, sign = self.tfsClf.predict(name)
		print(sign)
		# 预测完后就可以删除图片了
		os.remove(name)
		#cv2.imshow('frame', frame)
		cap.release()