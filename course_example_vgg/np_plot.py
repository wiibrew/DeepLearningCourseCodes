#the script is for geneate a figure for deep learning model 
#datablob and filter parameter visualization
import numpy as np 
import skimage

def plot_array(arr4d,size,stride=5):
	#
	_,h,w,d=arr4d.shape
	N_blocks=size/(h+stride)
	step=h+stride
	Im_arr=np.zeros((size,size))
	cnt=0
	for i in range(N_blocks):
		for j in range(N_blocks):
			Im_arr[i*step:(i+1)*step,j*step:(j+1)*step]=arr4d[0,:,:,cnt]
			cnt+=1
	return Im_arr
def plot_filter(arr4d,filter_num=6, stride=1):
	h=3
	N_blocks=filter_num
	step=4
	Im_arr=np.zeros((filter_num*step,filter_num*step))
	cnt=0
	for i in range(N_blocks):
		for j in range(N_blocks):
			Im_arr[i*step:(i+1)*step,j*step:(j+1)*step]=arr4d[:,:,0,cnt]
			cnt+=1
	skimage.transform.resize(img, (200, 200))
	return Im_arr