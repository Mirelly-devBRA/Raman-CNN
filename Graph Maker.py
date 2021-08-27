import matplotlib.pyplot as plt
import numpy as np
import glob

def plot_spectra(path):
    """Plots an individual spectra
    Inputs:
    path - path to the spectra you want to plot

    Output:
    A grapth
    """
    plt.figure(figsize=(20, 10))
    x, y= np.loadtxt(fname=path, delimiter='\t',dtype=int,
                      usecols = (1,2), skiprows=100, unpack = True)
    plt.plot(x, y)
    return plt.show()

def plot_folder(path):
    """Plots all of the spectra in a folder in a single graph
    Inputs:
    path - path to the folder which contains all the spectra

    Output:
    A grapth
    """
    plt.figure(figsize=(20, 10))
    for filename in glob.glob(path + '/*.pspec'):
        x, y= np.loadtxt(fname=filename,  delimiter='\t',dtype=int,  usecols = (1,2),
                          skiprows=100, unpack = True)
        plt.plot(x, y)
    return plt.show()

def heat_map(path):
    """Plots a spectra along with a heatmap of the spectra
    Inputs:
    path - path to the spectra
    
    Output:
    A grapth
    """
    x, y= np.loadtxt(fname=path, delimiter='\t',dtype=int,
                     usecols = (1,2), skiprows=100, unpack = True)

    fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True, figsize=(20,10))

    extent = [x[0]-(x[1]-x[0])/2, x[-1]+(x[1]-x[0])/2,0,1]
    ax.imshow(y[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax2.plot(x,y)
    plt.tight_layout()
    return plt.show()

path='C:/Users/david/Documents/ISI Placement/ISI/ISI_Dataset/Cetrizine_0.8s'
# plot_spectra(path)
for filename in glob.glob(path + '/*.pspec'):
    heat_map(filename)

# path = 'C:/Users/david/Documents/ISI Placement/ISI/ISI_Dataset'
# for folder in glob.glob(path+ '/*'):
#     plot_folder(folder)
    
