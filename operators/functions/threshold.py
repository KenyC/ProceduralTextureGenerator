from PIL import Image, ImageChops

def BWtoRGB(value):
	return (value,value,value)

def threshold(img, thres):
	return ImageChops.subtract(img,Image.new(img.mode,img.size,BWtoRGB(thres)), scale = (255.-thres)/255)



def colorRamp(img, markers):
	"""Markers is a list of (position, value) tuple, interpolation is linear"""
	markers.sort(key = lambda a,b: a)
	npArr = np.array(img)

	for i, (posI, valI) in enumerate(markers[:-1]):
		pass

