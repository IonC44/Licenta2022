# import the necessary packages
from keras.preprocessing.image import img_to_array

class ImageToArray:
	def __init__(self, dataFormat = None):
		# store the image data format
		self.dataFormat = dataFormat

	def preprocesare(self, image):
		# apply the Keras utility function that correctly rearranges
		# the dimensions of the image
        
		return img_to_array(image, data_format=self.dataFormat)
