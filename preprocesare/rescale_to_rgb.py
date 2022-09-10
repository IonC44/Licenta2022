import tensorflow as tf

class RescaleToRGB:
    def __init__(self, dim):
        self.dim = dim
        
    def preprocesare(self, image):
        image = tf.image.resize(images=image, size=(224, 224), method='bilinear').numpy() 
        imageRGB = tf.image.grayscale_to_rgb(tf.convert_to_tensor(image)).numpy()
        
        
        return imageRGB