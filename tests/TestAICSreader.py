from aicsimageio import imread, AICSImage
import numpy as np

fp = '/Volumes/comp_micro/rawdata/hummingbird/Cameron/20201029_MMreaderTest/StackTest_1/StackTest_1_MMStack.ome.tif'
img = AICSImage(fp)
#
print(np.arange(0,img.size_z,1))