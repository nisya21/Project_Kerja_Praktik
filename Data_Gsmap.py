# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:14:14 2021

@author: ASUS
"""

from mpl_toolkits.basemap import cm, Basemap
import os
import pylab
import struct
import numpy as np

#format data
format1='4320000f'
kolom = 3600 #jumlah longitude
baris = 1200 #jumlah latitude

lon = pylab.arange(0.,360.,0.1)
lat = pylab.arange(60.,-60.,-0.1)

#read/open file
namafile= r'C:\Users\ASUS\.spyder-py3\file\gsmap_gauge.20200101.0100\gsmap_gauge.20200101.0100.dat'
f=open(namafile,'rb')
dataset=f.read()
f.close()
prep = pylab.array(struct.unpack(format1,dataset)) #data rain rate

#rain <=0 -> nan
indx = np.where(prep<=0)
prep[indx[0]] = pylab.NaN
prep2d = prep.reshape(baris,kolom) #data rain rate 2 dimensi

#plot basemap
paralles=pylab.arange(-60.0,60.,20.)
meridians=pylab.arange(0.,180.,20)
xv,yv=pylab.meshgrid(lon,lat)
#clevs=pylab.arange(0,31,1);

m=Basemap(projection='cyl',llcrnrlat=-8,urcrnrlat=-5,
  llcrnrlon=105,urcrnrlon=109,resolution='h')
lon,lat = m(xv,yv)
m.drawcoastlines()
m.drawparallels(paralles,labels=[1,0,0,0],fontsize=10)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)

clev = pylab.arange(0,50,5)
cs = m.contourf(lon,lat,prep2d,clev,cmap='rainbow',extend='max')
cbar = m.colorbar(cs,location="right",pad="5%")
cbar.set_label("mm/hr")

pylab.title("GSMaP rain rate (mm/hr)")
#pylab.savefig("./gsmap.png")
pylab.show()