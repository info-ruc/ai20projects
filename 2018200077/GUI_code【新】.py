import wx
import os
import our_ocr
import time
from smoothnlp import kg
import cv2
import re
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from KG.kg import HarvestText
from rdflib import URIRef, Graph, Namespace, Literal
from pyxdameraulevenshtein import damerau_levenshtein_distance as edit_dis
import numpy as np
# matplotlib显示中文和负号问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from harvesttext import HarvestText
from KGQA import KGQA
PROJECT_PATH = '/Users/walden/Desktop/AI_Project'
#启动命令：/Users/walden/opt/anaconda3/bin/pythonw

class MyFrame(wx.Frame):    #创建自定义Frame
    def __init__(self, parent):
        wx.Frame.__init__(self, parent,id=-1,
                        title="MyOcr",size=(1300,665)) #设置窗体

        """
        panel和sizer是wxpython提供的窗口部件。是容器部件，可以用于存放其他窗口部件
        """
        self.res = ""
        self.doc = ""
        
        self.panel = wx.Panel(self)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel.SetSizer(self.sizer)
        self.filename = wx.TextCtrl(self, pos = (30,18),size = (220,25))
        self.contents = wx.TextCtrl(self, pos = (20,55),size = (500,550), style = wx.TE_MULTILINE | wx.HSCROLL)
        self.loadButton = wx.Button(self, label = '打开文件夹',pos = (300,15),size = (180,30))
        self.loadButton.Bind(wx.EVT_BUTTON, self.load)    #将按钮和load函数绑定
        
        self.Question_name = wx.TextCtrl(self, pos = (600,18),size = (220,25))
        self.Answer = wx.TextCtrl(self, pos = (600,55),size = (450,550), style = wx.TE_MULTILINE | wx.HSCROLL)
        self.Question_Button = wx.Button(self, label = '向我提问', pos = (850, 15), size = (180,30))
        self.Question_Button.Bind(wx.EVT_BUTTON, self.ask)    #将按钮和load函数绑定
        
        self.bitmap = wx.StaticBitmap(self, -1, pos = (550,10), size = (750, 650))
        #txt = wx.StaticText(panel,-1,"请输入图片名：")    #创建静态文本组件
        #sizer.Add(txt,0,400,wx.LEFT)
        self.Center()   #将窗口放在桌面环境的中间

    def ask(self, event):
        res_str = "".join(self.res)
        print(res_str)
        question = self.Question_name.GetValue()
        
        doc = self.res
        
        ht = HarvestText()
        sentences = ht.cut_sentences(doc)
        
        entity_type_dict = {}
        for i, sent in enumerate(sentences):
            entity_type_dict0 = ht.named_entity_recognition(sent)
            for entity0, type0 in entity_type_dict0.items():
                entity_type_dict[entity0] = type0
        for entity in list(entity_type_dict.keys())[:10]:
            print(entity, entity_type_dict[entity])
        ht.add_entities(entity_type_dict = entity_type_dict)
        inv_index = ht.build_index(sentences)
        counts = ht.get_entity_counts(sentences,inv_index)
        print(pd.Series(counts).sort_values(ascending=False).head())
        ht2 = HarvestText()
        SVOs = []
        for i, sent in enumerate(sentences):
            SVOs += ht2.triple_extraction(sent.strip())
        
        print("\n".join(" ".join(tri) for tri in SVOs[:]))
        fig = plt.figure(figsize=(12,8),dpi=100)
        g_nx = nx.DiGraph()
        labels = {}
        for subj, pred, obj in SVOs:
            g_nx.add_edge(subj,obj)
            labels[(subj,obj)] = pred
        pos=nx.spring_layout(g_nx)
        nx.draw_networkx_nodes(g_nx, pos, node_size=300)
        nx.draw_networkx_edges(g_nx,pos,width=4)
        nx.draw_networkx_labels(g_nx,pos,font_size=10,font_family='sans-serif')
        nx.draw_networkx_edge_labels(g_nx, pos, labels , font_size=10, font_family='sans-serif')
        plt.axis("off")
        
        QA = KGQA(SVOs, entity_type_dict=entity_type_dict)
        #questions = ["孙中山干了什么事？","清政府签订了哪些条约？","谁复辟了帝制？"]
        #for question0 in questions:
        #    print("问："+question0)
        #    print("答："+QA.answer(question0))
        print(QA.answer(question))
        self.Answer.SetValue(QA.answer(question))
        #plt.show()
        
    def load(self, event):
        print("load called.\n")
        reader = our_ocr.Reader(['ch_sim','en']) # need to run only once to load model into memory
        time_start = time.time()
        List = []
        load_output = ""
        address = self.filename.GetValue()
        os.chdir(address)
        
        for imgpath in os.listdir():
            print("正在读取："+imgpath)
            if imgpath.find('png') != -1:
                result = reader.readtext(imgpath)
                for line in result:
                    print(line)
                    strline = line[1]
                    strline = strline.replace('(','')
                    strline = strline.replace(')','')
                    strline = strline.replace('”','')
                    strline = strline.replace('“','')
                    strline = strline.replace(',','，')
                    strline = strline.replace('.','。')
                    strline = strline.replace('"','')
                    strline = strline.replace('夭','天')
                    strline = re.sub(r'[a-zA-Z]', '', strline)
                    if(strline.find('6')!=-1):
                        if(strline.find('6')<len(strline)-1):
                            if((strline[strline.find('6')-1] < '0' or strline[strline.find('6')-1] > '9') and (strline[strline.find('6')+1] < '0' or strline[strline.find('6')+1] > '9') ):
                                strline = strline.replace('6', '。')
                        else:
                            if(strline[strline.find('6')-1] < '0' or strline[strline.find('6')-1] > '9'):
                                strline = strline.replace('6', '。')
                    self.res = self.res + strline
                    List.append(strline)
                    load_output = load_output + strline + '\n'
                print(List)
        os.chdir('..')
        time_end = time.time()
        print(result)
        print("Time Cost: " + str(time_end - time_start) + ' s\n')
        
        
        #self.contents.SetValue(load_output)  #填充文本框
        self.contents.SetValue(self.doc)  #填充文本框
        
        
        '''
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
        '''
    
    
if __name__ == '__main__':
    os.chdir(PROJECT_PATH)
    app = wx.App()
    frame = MyFrame(None)   #为顶级窗口
    frame.Show(True)
    app.MainLoop()