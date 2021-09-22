import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
import geopandas as gpd
import osmnx as ox
import contextily as ctx
import pickle as pkl
import matplotlib.colors as colors
from tqdm import tqdm_notebook
from descartes import PolygonPatch
from shapely.geometry import Point,LineString,Polygon,MultiPolygon, geo
from numpy.core.fromnumeric import size
from collections import namedtuple

import tqdm





def seir(par, distr, flow, alpha, iterations, inf):
    
    r = flow.shape[0]
    n = flow.shape[1]
    N = distr[0].sum() # total population, we assume that N = sum(flow)
    
    Svec = distr[0].copy()
    Evec = np.zeros(n)
    Ivec = np.zeros(n)
    Rvec = np.zeros(n)
    
    if par.I0 is None:
        initial = np.zeros(n)
        # randomly choose inf infections
        for i in range(inf):
            loc = np.random.randint(n)
            if (Svec[loc] > initial[loc]):
                initial[loc] += 1.0
                
    else:
        initial = par.I0
    assert ((Svec < initial).sum() == 0)
    
    Svec =- initial
    Ivec =+ initial
    
    res = np.zeros((iterations, 5))
    res[0,:] = [Svec.sum(), Evec.sum(), Ivec.sum(), Rvec.sum(), 0]
    
    realflow = flow.copy() # copy!
    realflow = realflow / realflow.sum(axis=2)[:,:, np.newaxis]    
    realflow = alpha * realflow    
    
    history = np.zeros((iterations, 5, n))
    history[0,0,:] = Svec
    history[0,1,:] = Evec
    history[0,2,:] = Ivec
    history[0,3,:] = Rvec
    
    eachIter = np.zeros(iterations + 1)
    
    # run simulation
    for iter in range(0, iterations - 1):
        realOD = realflow[iter % r]
        
        d = distr[iter % r] + 1
        
        if ((d>N+1).any()): #assertion!
            print("Houston, we have a problem!")
            return res, history
        # N =  S + E + I + R
        
        newE = Svec * Ivec / d * par.R0 / par.DI
        newI = Evec / par.DE
        newR = Ivec / par.DI
        
        Svec -= newE
        Svec = (Svec 
               + np.matmul(Svec.reshape(1,n), realOD)
               - Svec * realOD.sum(axis=1)
                )
        Evec = Evec + newE - newI
        Evec = (Evec 
               + np.matmul(Evec.reshape(1,n), realOD)
               - Evec * realOD.sum(axis=1)
                )
                
        Ivec = Ivec + newI - newR
        Ivec = (Ivec 
               + np.matmul(Ivec.reshape(1,n), realOD)
               - Ivec * realOD.sum(axis=1)
                )
                
        Rvec += newR
        Rvec = (Rvec 
               + np.matmul(Rvec.reshape(1,n), realOD)
               - Rvec * realOD.sum(axis=1)
                )
                
        res[iter + 1,:] = [Svec.sum(), Evec.sum(), Ivec.sum(), Rvec.sum(), 0]
        eachIter[iter + 1] = newI.sum()
        res[iter + 1, 4] = eachIter[max(0, iter - par.HospiterIters) : iter].sum() * par.HospitalisationRate
        
        history[iter + 1,0,:] = Svec
        history[iter + 1,1,:] = Evec
        history[iter + 1,2,:] = Ivec
        history[iter + 1,3,:] = Rvec
        pass
    return res, history

def seir_plot(res):
    plt.plot(res[::12, 0], color='r', label='S')
    plt.plot(res[::12, 1], color='g', label='E')
    plt.plot(res[::12, 2], color='b', label='I')
    plt.plot(res[::12, 3], color='y', label='R')
    plt.plot(res[::12, 4], color='c', label='H')
    plt.legend()
    pass
#####################################
Param = namedtuple('Param', 'R0 DE DI I0 HospitalisationRate HospiterIters')
np.set_printoptions(suppress=True,precision=3)
pickle_OD_Metrices=open('COVID-19_DATA\Materials\Yerevan_OD_matrices.pkl','rb')
pickle_Population=open('COVID-19_DATA\Materials\Yerevan_population.pkl','rb')
OD_metrices=pkl.load(pickle_OD_Metrices)
popualtion=pkl.load(pickle_Population)
pickle_OD_Metrices.close()
pickle_Population.close()
r = OD_metrices.shape[0]
n = popualtion.shape[1]
N = 1000000.0
initialInd = [334, 353, 196, 445, 162, 297]
initial = np.zeros(n)
initial[initialInd] = 50
model = Param(R0=2.4, DE= 5.6 * 12, DI= 5.2 * 12, I0=initial, HospitalisationRate=0.1, HospiterIters=15*12)
alpha = np.ones(OD_metrices.shape)
iterations = 3000
res = {}
inf = 50
res['baseline'] = seir(model, popualtion, OD_metrices, alpha, iterations, inf)
seir_plot(res['baseline'][0])

###################################################
yerevane=gpd.read_file(r'COVID-19_DATA\Yerevan grid shapefile\Yerevan.shp')
yerevane.crs={"init":"epsg:4326"}
yerevane_3857=yerevane.to_crs(epsg=3857)
west,south,east,north=yerevane_3857.unary_union.bounds
baseline=res['baseline'][1][::12,:,:]
max_exp_idx=np.where(baseline[:,1,:]==baseline[:,1,:].max())[0].tolist()
max_exp_val=baseline[:,1,:].max()
##################### creating custom color map (transparent)
ncolor=150
color_array=plt.get_cmap('Reds')(range(ncolor))
color_array[:,-1]=np.linspace(0.2,1,ncolor)
color_array_r=color_array[::-1]
map_object=colors.LinearSegmentedColormap.from_list(name="my_reds",colors=color_array_r)
plt.register_cmap(cmap=map_object)
fig,ax2=plt.subplots()
h=ax2.imshow(np.random.rand(100,100),cmap="my_reds")
plt.colorbar(mappable=h)
#####################  ploting the final shape
params={"axes.labelcolor":"slategrey"}
plt.rcParams['axes.labelcolor']='slategrey'
for time in range(1,250):
    yerevane_3857['exposed']=baseline[time-1,0,:]
    fig,ax=plt.subplots(figsize=(14,14))
    yerevane_3857.loc[yerevane_3857.index==84,'exposed']=max_exp_val+1
    yerevane_3857.plot(ax=ax,facecolor="none",edgecolor='gray',alpha=0.5,linewidth=0.5,zorder=2)
    yerevane_3857.plot(ax=ax,column='exposed',cmap='my_reds',zorder=3)
    #
    ctx.add_basemap(ax,attribution="",url=ctx.sources._ST_TONER_LITE,zoom='auto',alpha=0.7)
    ax.set_xlim(west,east)
    ax.set_ylim(south,north)
    plt.tight_layout()
    #
    inset_ax=fig.add_axes([0.55,0.13,0.37,0.27])
    inset_ax.patch.set_alpha(0.5)
    inset_ax.plot(baseline[:time,0].sum(axis=1),label="susceptible",color='blue',ls="-",lw=1.5,alpha=0.8)
    inset_ax.plot(baseline[:time,1].sum(axis=1),label="exposed",color='y',ls="-",lw=1.5,alpha=0.8)
    inset_ax.plot(baseline[:time,2].sum(axis=1),label="infected",color='r',ls="-",lw=1.5,alpha=0.8)
    inset_ax.plot(baseline[:time,3].sum(axis=1),label="recovered",color='g',ls="-",lw=1.5,alpha=0.8)
    #
    inset_ax.scatter((time-1),baseline[(time-1),0].sum(),color="blue",s=50,alpha=0.5)
    inset_ax.scatter((time-1),baseline[(time-1),1].sum(),color="y",s=50,alpha=0.5)
    inset_ax.scatter((time-1),baseline[(time-1),2].sum(),color="r",s=50,alpha=0.5)
    inset_ax.scatter((time-1),baseline[(time-1),3].sum(),color="g",s=50,alpha=0.5)
    #
    inset_ax.scatter((time-1),baseline[(time-1),0].sum(),color="blue",s=20,alpha=0.5)
    inset_ax.scatter((time-1),baseline[(time-1),1].sum(),color="y",s=20,alpha=0.5)
    inset_ax.scatter((time-1),baseline[(time-1),2].sum(),color="r",s=20,alpha=0.5)
    inset_ax.scatter((time-1),baseline[(time-1),3].sum(),color="g",s=20,alpha=0.5)
    inset_ax.fill_between(np.arange(0,time),np.maximum(baseline[:time,0].sum(axis=1),baseline[:time,3].sum(axis=1)),alpha=0.034,color="r")
    inset_ax.plot([time,time],[0,max(baseline[(time-1,0)].sum(),baseline[(time-1,3)].sum())],ls='--',lw=0.7,alpha=0.8,color='r')
    inset_ax.grid(alpha=0.5)
    inset_ax.set_xlim(0,250)
    #
    inset_ax.set_ylabel("population",size=18,alpha=1,rotation=90)
    inset_ax.set_xlabel("Days",size=18,alpha=1)
    inset_ax.spines['right'].set_visible(False)
    inset_ax.spines['top'].set_visible(False)
    inset_ax.spines['left'].set_color('darkslategrey')
    inset_ax.spines['bottom'].set_color("darkslategrey")
    inset_ax.tick_params(axis='x',color='darkslategrey')
    inset_ax.tick_params(axis='y',color='darkslategrey')
    plt.legend(prop={'size':14,'weight':'light'},framealpha=0.5)
    plt.title("Yerevan Covid-19 Forcast",fontsize=18,color='dimgray')
    plt.savefig("plot/shape{}.jpg".format(time))
    
from os import listdir
import re
import imageio
from tqdm import tqdm_notebook
def sort_in_order( l ):    
    convert = lambda text: int(text) if text.isdigit() else text
    alphanumeric_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanumeric_key)
    pass
foldername = listdir("plot/")
filenames = sort_in_order(foldername)
with imageio.get_writer('Covid_19.gif', mode='I', fps=16) as writer:
    for filename in tqdm_notebook(filenames):
        image = imageio.imread('plot/{}'.format(filename))
        writer.append_data(image)
