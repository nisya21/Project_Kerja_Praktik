# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 19:44:58 2021

@author: ASUS
"""

#import conda
import os
import matplotlib

#membuka data netCDF (*.nc)
from netCDF4 import Dataset
file = './NC_H08_20200101_0020_L2CLP010_FLDK.02401_02401.nc'
dset = Dataset(file,'r')

#mengakses data/parameter/variables
lats = dset.variables['latitude'][:]
lons = dset.variables['longitude'][:]
cler = dset.variables['CLER_23'][:] 
clot = dset.variables['CLOT'][:] 
clth = dset.variables['CLTH'][:]
cltt = dset.variables['CLTT'][:]
cltype = dset.variables['CLTYPE'][:]

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
paralles = pylab.arange(-90.0,90.,1)
meridians = pylab.arange(0.,360.,1)
m.drawcoastlines(color='k',linewidth=0.3)
m.drawparallels(paralles,labels=[1,0,0,0],fontsize=8,color='k',linewidth=0.2)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=8,color='k',linewidth=0.2)

#filled contour
xv,yv=pylab.meshgrid(lons,lats)
lon,lat = m(xv,yv)

# plot data
clev = pylab.arange(1,10,1) 
colors = ['#6497b1','#005b96','#011f4b','#eae374',
          '#f9d62e','#fc913a','#fe8181','#fe2e2e','#b62020']
#label = ['Ci','Cs','DC','Ac','As','Ns','Cu','Sc','St']
cs = m.contourf(lon,lat,cltt,colors=colors)
cbar = m.colorbar(cs,location='right',pad='5%')
cbar.set_label("kelvin")

import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('10')
patch=[]
for i in colors:
    patch.append(mpatches.Patch(color=i))
#pylab.legend(patch,label,loc=2,prop=fontP
   #        ,borderaxespad=0.,bbox_to_anchor=(1.02,1))

#garis pantai
m.drawcoastlines(color='k',linewidth=0.3)

#map title
#pylab.title('Cloud Effective Radius Using Band 6')
#pylab.title('Cloud Optical Thickness')
#pylab.title('Cloud Top Height')
pylab.title('Cloud Top Temperature')
#pylab.title('Cloud Type Under ISCCP Cloud Type Classification Definition')

#save figure
#pylab.savefig('Himawari.png')

#show figure
pylab.show()
