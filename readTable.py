# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:52:00 2019

@author: Administrator
"""

import xlrd #导入x1rd库
data=xlrd.open_workbook('data.xlsx') #打开Exce1文件
sh=data.sheet_by_name('Sheet1') #获得需要的表单
print(sh.cell_value(1,1)) #打印表单中B2值
Xlst=[]
ylst=[]
for i in range(1, 126):
    singleXList=[float(sh.cell_value(i,1)),float(sh.cell_value(i,2)),float(sh.cell_value(i,3))]
    Xlst.append(singleXList)
    singleyList=[float(sh.cell_value(i,3))]
    ylst.append(singleyList)