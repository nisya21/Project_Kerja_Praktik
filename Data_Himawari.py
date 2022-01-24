# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 22:53:32 2021

@author: ASUS
"""

import matplotlib
import os

#membuka data netCDF (*.nc)
from netCDF4 import Dataset
file = './NC_H08_20200101_0010_R21_FLDK.02401_02401.nc'
dset = Dataset(file,'r')

#dimensi dan variabel
#print = dset.dimensions.keys()
#print = dset.variables.keys()
odict_keys = (['latitude', 'longitude', 'band_id', 'start_time', 'end_time', 
'geometry_parameters', 'albedo_01', 'albedo_02', 'albedo_03', 
'sd_albedo_03', 'albedo_04', 'albedo_05', 'albedo_06', 'tbb_07', 'tbb_08', 
'tbb_09', 'tbb_10', 'tbb_11', 'tbb_12', 'tbb_13', 'tbb_14', 'tbb_15', 'tbb_16', 
'SAZ', 'SAA', 'SOZ', 'SOA', 'Hour'])

#mengakses data/parameter/variables
lats = dset.variables['latitude'][:]
lons = dset.variables['longitude'][:]
b01 = dset.variables['albedo_01'][:] 
b08 = dset.variables['tbb_08'][:] 
b13 = dset.variables['tbb_13'][:]
b15 = dset.variables['tbb_15'][:]
b16 = dset.variables['tbb_16'][:]

#cek data
#print = lats; 
#print = b01; 
#print = b13;
import numpy as np
np.shape; np.max, np.min; np.mean

######## MAP PLOTTING ########
#import modul
proj_lib = os.path.join(os.path.join('Library'), 'share')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap
import pylab

#setup map projection
m = Basemap(projection='cyl',llcrnrlat=-8,urcrnrlat=-5,
        llcrnrlon=105,urcrnrlon=109,resolution='h')

#menampilkan garis lintang-bujur
paralles = pylab.arange(-90.0,90.,5)
meridians = pylab.arange(0.,360.,10)
m.drawparallels(paralles,labels=[1,0,0,0],fontsize=8,color='k',linewidth=0.2)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=8,color='k',linewidth=0.2)

#filled contour
xv,yv=pylab.meshgrid(lons,lats)
lon,lat = m(xv,yv)
clev = pylab.arange(0,1,0.05)
cs = m.contourf(lon,lat,b01) #clev,cmap='Greys_r

#legend
cbar = m.colorbar(cs,location='right',pad='5%')

#garis pantai
m.drawcoastlines(color='k',linewidth=0.3)

#map title
pylab.title('albedo_01')

#save figure
#pylab.savefig('albedo_01.png')

#show figure
pylab.show()
