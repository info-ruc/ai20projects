import numpy as np
from os import listdir
from sklearn.externals import joblib
from functools import reduce
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
 
path = 'D://Tello/img3'
model_path = "D://Tello/model3/"
test_path = "D://Tello/test"
 
test_accuracy = []
 
#读txt文件并将每个文件的描述子改为一维的矩阵存储
def txtToVector(filename, N):
    returnVec = np.zeros((1,N))
    fr = open(filename)
    lineStr = fr.readlines()
    i = 0
    for lines in lineStr:
        line = lines.split(' ')
        returnVec[0, i] = int(line[0])
        i = i+1
    return returnVec

def tran_SVM(N):
    svc = SVC()
    parameters = {'kernel':('linear', 'rbf'),
	              'C':[1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
	              'gamma':[0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
                  'max_iter':[100000,1000000]}#预设置一些参数值
    hwLabels = []#存放类别标签
    trainingFileList = listdir(path)
    #print(trainingFileList)
    m = int(len(trainingFileList))
    trainingMat = np.zeros((m*20,N))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        #print(fileNameStr)
        classNumber = int(fileNameStr.split('_')[0])
        for j in range(1,21):
            hwLabels.append(classNumber)
            trainingMat[i*20+j-1 ,:] = txtToVector(path+'/'+fileNameStr+ '/' + str(j)+'.txt',N)#将训练集改为矩阵格式
    print("数据加载完成")
    clf = GridSearchCV(svc, parameters, cv=5, n_jobs=8)#网格搜索法，设置5-折交叉验证
    clf.fit(trainingMat,hwLabels)
    print(clf.return_train_score)
    print(clf.best_params_)#打印出最好的结果
    best_model = clf.best_estimator_
    print("SVM Model save...")
    save_path = model_path + "svm_efd_" + "train_model.m"
    joblib.dump(best_model,save_path)#保存最好的模型

def test_SVM(clf,N):
    testFileList = listdir(test_path)
    errorCount = 0#记录错误个数
    #mTest = len(testFileList)
    mTest = 96*10
    for testfile in testFileList:
        #fileNameStr = testFileList[i]
        for tmp in range(96):
            classNum = int(testfile)
            vectorTest = txtToVector(test_path + '/' + testfile + '/' + str(tmp) + '.txt',N)
            valTest = clf.predict(vectorTest)
            print("分类返回结果为%d\t真实结果为%d" % (valTest, classNum))
            if valTest != classNum:
                errorCount += 1
            tmp = tmp + 1
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))

if __name__ == '__main__':
    tran_SVM(31)
    clf = joblib.load(model_path + "svm_efd_" + "train_model.m")
    #test_SVM(clf,31)

'''
    parameters = {'kernel':('linear', 'rbf'),
	              'C':[1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
	              'gamma':[0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]}#预设置一些参数值
'''