# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:13:14 2021

@author: ASUS
"""

# Load libraries
import os
import netCDF4
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset


# Load dataset
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"

#datatraining
file1 = './NC_H08_20200101_0010_R21_FLDK.02401_02401.nc'
file2 = './NC_H08_20200101_0010_L2CLP010_FLDK.02401_02401.nc'
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

dset1 = Dataset(file1,'r')
dset2 = Dataset(file2,'r')

lats = dset1.variables['latitude'][:]
lons = dset1.variables['longitude'][:]

b07 = (ma.getdata(dset1.variables['tbb_07'][:])).ravel()
b08 = (ma.getdata(dset1.variables['tbb_08'][:])).ravel()
b09 = (ma.getdata(dset1.variables['tbb_09'][:])).ravel()
b10 = (ma.getdata(dset1.variables['tbb_10'][:])).ravel()
b11 = (ma.getdata(dset1.variables['tbb_11'][:])).ravel()
b12 = (ma.getdata(dset1.variables['tbb_12'][:])).ravel()
b13 = (ma.getdata(dset1.variables['tbb_13'][:])).ravel()
b14 = (ma.getdata(dset1.variables['tbb_14'][:])).ravel()
b15 = (ma.getdata(dset1.variables['tbb_15'][:])).ravel()
b16 = (ma.getdata(dset1.variables['tbb_16'][:])).ravel()

cler = (dset2.variables['CLER_23'][:]).ravel()
clot = (dset2.variables['CLOT'][:]).ravel()
clth = (dset2.variables['CLTH'][:]).ravel()
cltt = (ma.getdata(dset2.variables['CLTT'][:])).ravel()
cltype = (dset2.variables['CLTYPE'][:]).ravel()

#datavalidasi
file3 = './NC_H08_20200101_0020_R21_FLDK.02401_02401.nc'
file4 = './NC_H08_20200101_0020_L2CLP010_FLDK.02401_02401.nc'

dset3 = Dataset(file3,'r')
dset4 = Dataset(file4,'r')

b07v = (ma.getdata(dset3.variables['tbb_07'][:])).ravel()
b08v = (ma.getdata(dset3.variables['tbb_08'][:])).ravel()
b09v = (ma.getdata(dset3.variables['tbb_09'][:])).ravel()
b10v = (ma.getdata(dset3.variables['tbb_10'][:])).ravel()
b11v = (ma.getdata(dset3.variables['tbb_11'][:])).ravel()
b12v = (ma.getdata(dset3.variables['tbb_12'][:])).ravel()
b13v = (ma.getdata(dset1.variables['tbb_13'][:])).ravel()
b14v = (ma.getdata(dset1.variables['tbb_14'][:])).ravel()
b15v = (ma.getdata(dset3.variables['tbb_15'][:])).ravel()
b16v = (ma.getdata(dset3.variables['tbb_16'][:])).ravel()

clerv = (dset4.variables['CLER_23'][:]).ravel()
clotv = (dset4.variables['CLOT'][:]).ravel()
clthv = (dset4.variables['CLTH'][:]).ravel()
clttv = (ma.getdata(dset4.variables['CLTT'][:])).ravel()
cltypev = (dset4.variables['CLTYPE'][:]).ravel()

# Split-out validation dataset
from sklearn import model_selection

X = []
for i in range(len(b07)):
    X.append([b08[i],b13[i],b15[i]])

Y = []
for i in range(len(cltt)):
    if cltt[i] >= 180 and cltt[i] < 190:
        Y.append(1)
    elif cltt[i] >= 190 and cltt[i] < 200:
        Y.append(2)
    elif cltt[i] >= 200 and cltt[i] < 210:
        Y.append(3)
    elif cltt[i] >= 210 and cltt[i] < 220:
        Y.append(4)
    elif cltt[i] >= 220 and cltt[i] < 230:
        Y.append(5)
    elif cltt[i] >= 230 and cltt[i] < 240:
        Y.append(6)
    elif cltt[i] >= 240 and cltt[i] < 250:
        Y.append(7)
    elif cltt[i] >= 250 and cltt[i] < 260:
        Y.append(8)
    elif cltt[i] >= 260 and cltt[i] < 270:
        Y.append(9)
    elif cltt[i] >= 270 and cltt[i] < 280:
        Y.append(10)
    elif cltt[i] >= 280 and cltt[i] < 290:
        Y.append(11)
    elif cltt[i] >= 290 and cltt[i] < 300:
        Y.append(12)
    else : 
        Y.append(13)
        
        
Y = np.array (Y).reshape(5764801,1)


X_train=X[:]
Y_train=Y[:]


Xv = []
for i in range(len(b07)):
    Xv.append([b08v[i],b13v[i],b15v[i]])

Yv = []
for i in range(len(cltt)):
    if clttv[i] >= 180 and clttv[i] < 190:
        Yv.append(1)
    elif clttv[i] >= 190 and clttv[i] < 200:
        Yv.append(2)
    elif clttv[i] >= 200 and clttv[i] < 210:
        Yv.append(3)
    elif clttv[i] >= 210 and clttv[i] < 220:
        Yv.append(4)
    elif clttv[i] >= 220 and clttv[i] < 230:
        Yv.append(5)
    elif clttv[i] >= 230 and clttv[i] < 240:
        Yv.append(6)
    elif clttv[i] >= 240 and clttv[i] < 250:
        Yv.append(7)
    elif clttv[i] >= 250 and clttv[i] < 260:
        Yv.append(8)
    elif clttv[i] >= 260 and clttv[i] < 270:
        Yv.append(9)
    elif clttv[i] >= 270 and clttv[i] < 280:
        Yv.append(10)
    elif clttv[i] >= 280 and clttv[i] < 290:
        Yv.append(11)
    elif clttv[i] >= 290 and clttv[i] < 300:
        Yv.append(12)
    else : 
        Yv.append(13)
        
Yv = np.array (Yv).reshape(5764801,1)

X_validation=Xv[:]
Y_validation=Yv[:]


# Build models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

models = []
#models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVR(gamma='auto')))
#models.append(('RF',RandomForestClassifier(n_estimators=10)))
#models.append(('ETC',ExtraTreesClassifier(n_estimators=10)))
#models.append(('MLP',MLPClassifier()))

Y_prediction=[]
names=[]
for name, model in models:
        model.fit(X_train,Y_train.ravel())
        prediction=model.predict(X_validation)
        Y_prediction.append(prediction)
        names.append(name)

#Accuracy score
from sklearn.metrics import accuracy_score

acc=[]
for i in range (len(Y_prediction)):
        accuracy = accuracy_score(Y_validation, Y_prediction[i])
        acc.append(accuracy)
        
for i in range (len(acc)):
        print (names[i],'=',acc[i])

######## MAP PLOTTING ########
#import modul
proj_lib = os.path.join(os.path.join('Library'), 'share')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap
import pylab

Y_prediction = np.array(Y_prediction)
Y_prediction = 170+(10*Y_prediction)

yp = np.array (Y_prediction).reshape(2401,2401)
#setup map projection
#m = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=20,
        #llcrnrlon=90,urcrnrlon=150,resolution='h')

#setup map projection
m = Basemap(projection='cyl',llcrnrlat=-8,urcrnrlat=-5,
        llcrnrlon=105,urcrnrlon=109,resolution='h')

#menampilkan garis lintang-bujur
paralles = pylab.arange(-90.0,90.,5)
meridians = pylab.arange(0.,360.,10)
m.drawcoastlines(color='k',linewidth=0.3)
m.drawparallels(paralles,labels=[1,0,0,0],fontsize=8,color='k',linewidth=0.2)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=8,color='k',linewidth=0.2)

#filled contour
xv,yv=pylab.meshgrid(lons,lats)
lon,lat = m(xv,yv)

# plot data
clev = pylab.arange(180,301,10) 

label = ['Ci','Cs','DC','Ac','As','Ns','Cu','Sc','St']
cs = m.contourf(lon,lat,yp,levels=clev)
cbar = m.colorbar(cs,location='right',pad='5%')

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