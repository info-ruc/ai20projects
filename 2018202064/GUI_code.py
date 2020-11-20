import wx
import os
import easyocr
import time
from smoothnlp import kg
import cv2
PROJECT_PATH = '/Users/walden/Desktop/AI_Project'
#启动命令：/Users/walden/opt/anaconda3/bin/pythonw
"""
Every wx application must have a single ``wx.App`` instance, and all
    creation of UI objects should be delayed until after the ``wx.App`` object
    has been created in order to ensure that the gui platform and wxWidgets
    have been fully initialized.
"""
class MyFrame(wx.Frame):    #创建自定义Frame
    def __init__(self, parent):
        wx.Frame.__init__(self, parent,id=-1,
                        title="MyOcr",size=(1300,665)) #设置窗体

        """
        panel和sizer是wxpython提供的窗口部件。是容器部件，可以用于存放其他窗口部件
        """
        self.panel = wx.Panel(self)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel.SetSizer(self.sizer)
        self.filename = wx.TextCtrl(self, pos = (30,18),size = (220,25))
        self.contents = wx.TextCtrl(self, pos = (20,55),size = (500,550), style = wx.TE_MULTILINE | wx.HSCROLL)
        self.loadButton = wx.Button(self, label = '打开图片(png format)',pos = (300,15),size = (180,30))
        self.loadButton.Bind(wx.EVT_BUTTON, self.load)    #将按钮和load函数绑定
        self.bitmap = wx.StaticBitmap(self, -1, pos = (550,10), size = (750, 650))
        #txt = wx.StaticText(panel,-1,"请输入图片名：")    #创建静态文本组件
        #sizer.Add(txt,0,400,wx.LEFT)
        self.Center()   #将窗口放在桌面环境的中间

    def load(self, event):
        time_start = time.time()
        address = self.filename.GetValue()
        reader = easyocr.Reader(['ch_sim','en']) # need to run only once to load model into memory
        result = reader.readtext(address)
        time_end = time.time()
        #print(result)
        res = "Time Cost: " + str(time_end - time_start) + ' s\n'
        List = []
        for line in result:
            res = res + line[1] + '\n'
            List.append(line[1])
        print(List)
        self.contents.SetValue(res)  #填充文本框
        rels = kg.extract(text=List)    
        g = kg.rel2graph(rels)  ## 依据文本解析结果, 生成networkx有向图
        fig = kg.graph2fig(g, x=1500, y=1500)  ## 生成 matplotlib.figure.Figure 图片
        fig.savefig("KG_test_output.png")
        img = cv2.imread("KG_test_output.png")
        img = img[200:1500, :, :]
        cv2.imwrite("KG_test_output.png", img)
        img = wx.Image("KG_test_output.png", wx.BITMAP_TYPE_PNG).Rescale(750, 650)
        img = img.ConvertToBitmap()
        self.bitmap.SetBitmap(img)
    
    
    
if __name__ == '__main__':
    os.chdir(PROJECT_PATH)
    app = wx.App()
    frame = MyFrame(None)   #为顶级窗口
    frame.Show(True)
    app.MainLoop()