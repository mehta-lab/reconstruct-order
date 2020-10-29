from aicsimageio import imread, AICSImage

fp = '/Volumes/comp_micro/rawdata/hummingbird/Cameron/20201029_MMreaderTest/StackTest_1/StackTest_1_MMStack.ome.tif'
img = AICSImage(fp)
#
print(img.metadata)