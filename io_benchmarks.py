import os
import threading
import multiprocessing
import zarr
import numpy as np
import z5py
import h5py
import tifffile
import dask.array as da
import xarray as xr
import time as timer
from numcodecs import GZip, BZ2
from shutil import rmtree

def write_speed(t1, t2):

	print('Write speed: ' + str(data.size*2/10**6/(t2-t1)) + ' MB/sec')

def write_tiff_data(idx, data, blockSize):
    
    for i in range(idx, idx+blockSize):
    	tifffile.imsave('C:\\benchmark\\tiff\\' + str(i) + '.tiff', data[i])

def write_data(idx, zdata, data, blockSize):
    
    zdata[idx:idx+blockSize] = data[idx:idx+blockSize]
    
def write_zarr(data, chunks, compressor, parallel, mode):
    
    if mode == 'n5':
        store = zarr.N5Store('C:\\benchmark\\zarr.n5')
    else:
        store = zarr.DirectoryStore('C:\\benchmark\\zarr.zr')
    
    t1=timer.time()

    blockSize = 2*chunks[0]
    
    zdata = zarr.zeros(store = store, overwrite = True, shape = data.shape, chunks = chunks, dtype = data.dtype, compressor = compressor)
    
    if parallel == True:
        
        threads = []

        for idx in np.arange(0, data.shape[0], blockSize):
            thread = threading.Thread(target = write_data, args=(idx, zdata, data, blockSize))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()
                                      
    else:
                                      
        zdata[:] = data[:]

    t2=timer.time()
    
    write_speed(t1,t2)

def write_zarr_xarray(data, chunks, compressor, chunked, mode):
    
    if mode == 'n5':
        store = zarr.N5Store('C:\\benchmark\\zarr_xarray.n5')
    else:
        store = zarr.DirectoryStore('C:\\benchmark\\zarr_xarray.zr')
    
    t1=timer.time()

    blockSize = 2*chunks[0]
    
    darray = da.from_array(data, chunks = chunks)
    xdataset = xr.Dataset({'i': (('z','x','y'), darray)})
    xdataset.to_zarr(store, compute=False, encoding={'i': {'compressor': compressor}})
    
    if chunked == True:
    
        for idx in range(0, data.shape[0], blockSize):
            selection = {'z': slice(idx, idx + blockSize)}
            xdataset.isel(selection).to_zarr(store, region = selection)
            
    else:
        
        xdataset.to_zarr(store, encoding={'i': {'compressor': compressor}})

    t2=timer.time()
    
    write_speed(t1,t2)

def write_zarr_dask(data, chunks, compressor, mode):
    
    if mode == 'n5':
        store = zarr.N5Store('C:\\benchmark\\zarr_dask.n5')
    else:
        store = zarr.DirectoryStore('C:\\benchmark\\zarr_dask.zr')
    
    t1=timer.time()

    darray = da.from_array(data, chunks = chunks)

    darray.to_zarr(store, compressor = compressor)

    t2=timer.time()
    
    write_speed(t1,t2)

def write_z5py(data, chunks, compressor, parallel, mode):
    
    if mode == 'n5':
        f = z5py.File('C:\\benchmark\\z5py.n5', use_zarr_format=False)
    else:
        f = z5py.File('C:\\benchmark\\z5py.zr', use_zarr_format=True)

    t1=timer.time()

    blockSize = 2*chunks[0]
    
    zdata = f.create_dataset('test', shape=data.shape, chunks=chunks, dtype=data.dtype, compression=compressor)

    if parallel == True:
        
        threads = []

        for idx in np.arange(0, data.shape[0], blockSize):
            thread = threading.Thread(target = write_data, args=(idx, zdata, data, blockSize))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()
                                      
        else:
            
            zdata[:] = data[:]

    t2=timer.time()

    write_speed(t1,t2)

def write_h5py(data, chunks, compressor):
    
    t1=timer.time()

    f = h5py.File('C:\\benchmark\\h5py.h5','w')
    zdata = f.create_dataset('test', shape = data.shape, chunks = chunks, dtype = data.dtype, compression = compressor)
    zdata[:] = data
    
    t2=timer.time()
    
    write_speed(t1,t2)

def write_tiff(data, blockSize, parallel):

	t1=timer.time()

	if parallel == True:

		os.mkdir('C:\\benchmark\\tiff')
		threads = []

		for idx in np.arange(0, data.shape[0], blockSize):
			thread = threading.Thread(target = write_tiff_data, args=(idx, data, blockSize))
			thread.start()
			threads.append(thread)

		for t in threads:
			t.join()

	else:

		tifffile.imsave('C:\\benchmark\\tiff.tiff', data, bigtiff = True)

	t2=timer.time()

	write_speed(t1,t2)

if __name__ == '__main__':

	cores = multiprocessing.cpu_count()
	
	chunks = (32, 256, 256)
	data = np.random.randint(0, 2000, size = [cores*chunks[0],2048,2048]).astype('uint16')

	if os.path.exists('C:\\benchmark'):
		rmtree('C:\\benchmark')
	os.mkdir('C:\\benchmark')

	write_z5py(data, chunks = chunks, compressor = 'raw', parallel = True, mode = 'zarr')
	write_z5py(data, chunks = chunks, compressor = 'raw', parallel = True, mode = 'n5')
	write_zarr(data, chunks = chunks, compressor = None, parallel = True, mode = 'zarr')
	write_zarr(data, chunks = chunks, compressor = None, parallel = True, mode = 'n5')
	write_zarr_dask(data, chunks = chunks, compressor = None, mode = 'zarr')
	write_zarr_dask(data, chunks = chunks, compressor = None, mode = 'n5')
	write_zarr_xarray(data, chunks = chunks, compressor = None, chunked = True, mode = 'zarr')
	write_zarr_xarray(data, chunks = chunks, compressor = None, chunked = True, mode = 'n5')
	write_h5py(data, chunks = chunks, compressor = None)
	write_tiff(data, blockSize = 64, parallel = True)
	write_tiff(data, blockSize = 64, parallel = False)