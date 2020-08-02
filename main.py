# -*- coding: utf-8 -*- 

import wx
import os
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import numpy as np

r_name = ' '
g_name = ' '
b_name = ' '
img_name = ' '

#부모의 frame을 가져와서 메뉴나 부가기능 추가
class UnnamedYet(wx.Frame):
    
    def __init__(self, *argv, **kw):
        # ensure the parent's __init__ is called
        super(UnnamedYet, self).__init__(*argv, **kw)

        # create a panel in the frame
        self.pnl = wx.Panel(self)
        
        copyright = "COPYRIGHT ⓒ 2020 EME(Explore to Exo-planet) ALL RIGHTS RESERVED."
        font = wx.Font(10, wx.DEFAULT, wx.ITALIC, wx.NORMAL)
        st = wx.StaticText(self.pnl, -1, copyright, (0, 105), style=wx.ALIGN_CENTRE)
        st.SetFont(font)
        
        self.makeControl()
        self.makeMenuBar()
        self.CreateStatusBar()
        self.SetStatusText("Welcome!")
        
    def makeControl(self):
        
        #self.txt = wx.TextCtrl(self.pnl, -1, size=(140, -1)) #입력할 수 있는 INPUT BOX/ -1은 default
        #self.txt.SetValue("This is placeholder..")
        
        self.btn = wx.Button(self.pnl, -1, "Name Setting")
        self.Bind(wx.EVT_BUTTON, self.GetName, self.btn)
        self.txt = wx.TextCtrl(self.pnl, -1, size=(140,-1))
        self.txt.SetValue('.png')
        
        #버튼 세 개
        self.btnFile_r = wx.Button(self.pnl, wx.ID_ANY, 'Open File R')
        self.Bind(wx.EVT_BUTTON, self.openFile_r, self.btnFile_r)
        
        self.btnFile_g = wx.Button(self.pnl, wx.ID_ANY, 'Open File G')
        self.Bind(wx.EVT_BUTTON, self.openFile_g, self.btnFile_g)
        
        self.btnFile_b = wx.Button(self.pnl, wx.ID_ANY, 'Open File B')
        self.Bind(wx.EVT_BUTTON, self.openFile_b, self.btnFile_b)
        
        self.mergebtn = wx.Button(self.pnl, wx.ID_ANY, 'Merge') #버튼 만들기
        self.Bind(wx.EVT_BUTTON, self.mergeButton, self.mergebtn)
        
        self.st1 = wx.StaticText(self.pnl, label="Merge Celestial RGB Images!!!")
        font1 = self.st1.GetFont() #st=StaticText
        font1.PointSize += 10
        font1 = font1.Bold()
        self.st1.SetFont(font1)
        
        sizer = wx.BoxSizer(wx.VERTICAL) #sizer = 크기가 바껴도 자동으로 비율 조정
        sizer.Add(self.st1)
        
        sizer.Add(self.btn)
        sizer.Add(self.txt)
        
        sizerButtons = wx.WrapSizer(wx.HORIZONTAL)
        sizerButtons.Add(self.btnFile_r) #r 가져오는 버튼 페널에 나타내기
        sizerButtons.Add(self.btnFile_g) #g 가져오는 버튼 페널에 나타내기
        sizerButtons.Add(self.btnFile_b) #b 가져오는 버튼 페널에 나타내기
        sizerButtons.Add(self.mergebtn) #merge 하는 버튼 페널에 나타내기
        sizer.Add(sizerButtons)
        self.pnl.SetSizer(sizer)        
        
    #이름 설정
    def GetName(self, e):

        dlg = wx.TextEntryDialog(self.pnl, '.png 형식만 지원:',"사진 이름 설정","", 
                style=wx.OK)
        dlg.ShowModal()
        global img_name
        img_name = dlg.GetValue()
        self.txt.SetValue(img_name)
        dlg.Destroy()
    
    # 파일 오픈 다이얼로그 띄우기(이벤트 처리기)
    def openFile_r(self,  event):
        # 파일 오픈 다이얼로그 생성
        dlg = wx.FileDialog(self, "Choose a file", os.getcwd(), "", "*.*", wx.FD_OPEN)

       # 파일 다이얼로그에서 파일을 선택후 ok 버튼을 누르면
        if dlg.ShowModal() == wx.ID_OK:
            # 파일의 경로를 구함
            r = dlg.GetPath()
            # 파일 경로에서 파일명만 추출함
            mypath = os.path.basename(r)
            # 선택한 파일의 이름을 Frame의 상태바에 출력함
            self.SetStatusText("You selected: %s" % mypath)

            global r_name
            r_name = r
       # 파일 오픈 다이얼로그 파괴
        dlg.Destroy()
        
    # 파일 오픈 다이얼로그 띄우기(이벤트 처리기)
    def openFile_g(self, event):
        # 파일 오픈 다이얼로그 생성
        dlg = wx.FileDialog(self, "Choose a file", os.getcwd(), "", "*.*", wx.FD_OPEN)
       # 파일 다이얼로그에서 파일을 선택후 ok 버튼을 누르면
        if dlg.ShowModal() == wx.ID_OK:
            # 파일의 경로를 구함
            g = dlg.GetPath()
            # 파일 경로에서 파일명만 추출함
            mypath = os.path.basename(g)
            # 선택한 파일의 이름을 Frame의 상태바에 출력함
            self.SetStatusText("You selected: %s" % mypath)
            global g_name
            g_name = g

       # 파일 오픈 다이얼로그 파괴
        dlg.Destroy()
        
    # 파일 오픈 다이얼로그 띄우기(이벤트 처리기)
    def openFile_b(self,  event):
        # 파일 오픈 다이얼로그 생성
        dlg = wx.FileDialog(self, "Choose a file", os.getcwd(), "", "*.*", wx.FD_OPEN)
       # 파일 다이얼로그에서 파일을 선택후 ok 버튼을 누르면
        if dlg.ShowModal() == wx.ID_OK:
            # 파일의 경로를 구함
            b = dlg.GetPath()
            # 파일 경로에서 파일명만 추출함
            mypath = os.path.basename(b)
            # 선택한 파일의 이름을 Frame의 상태바에 출력함
            self.SetStatusText("You selected: %s" % mypath)
            global b_name
            b_name = b

       # 파일 오픈 다이얼로그 파괴
        dlg.Destroy()

    def mergeButton(self, evt):
        global r_name
        global g_name
        global b_name
        
        r0 = fits.open(r_name)[0].data - 255
        g0 = fits.open(g_name)[0].data - 255
        b0 = fits.open(b_name)[0].data - 255

        r = r0.astype(float)
        g = g0.astype(float)
        b = b0.astype(float)

        rgb = make_lupton_rgb(r, g, b, Q=10, stretch=0.1, filename=img_name)
        plt.imshow(rgb, origin='lower')
        plt.waitforbuttonpress()
        
    def makeMenuBar(self):

        #메뉴1
        fileMenu = wx.Menu()
        helloItem = fileMenu.Append(-1, "&Hello...\tCtrl-H", #이름: Hello... 단축키:Ctrl + H
                "Help string shown in status bar for this menu item")
        fileMenu.AppendSeparator() #나누는 줄
        exitItem = fileMenu.Append(wx.ID_EXIT) #종료하는 부분

        #메뉴2
        helpMenu = wx.Menu()
        aboutItem = helpMenu.Append(wx.ID_ABOUT)

        #메뉴1 메뉴2 올리기
        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu, "&File")
        menuBar.Append(helpMenu, "&Help")

        self.SetMenuBar(menuBar)

        #버튼 누른 후 event 설정
        self.Bind(wx.EVT_MENU, self.OnHello, helloItem)
        self.Bind(wx.EVT_MENU, self.OnExit,  exitItem)
        self.Bind(wx.EVT_MENU, self.OnAbout, aboutItem)

    def OnExit(self, event):
        self.Destroy()

    def OnHello(self, event):
        wx.MessageBox("Hello again from wxPython")

    def OnAbout(self, event):
        wx.MessageBox("This is a wxPython Hello World sample",
                      "About Hello World 2",
                      wx.OK|wx.ICON_INFORMATION)
        
app = wx.App()
frm = UnnamedYet(None, title='Unnamed', size=(500,200))
frm.Show()
app.MainLoop()