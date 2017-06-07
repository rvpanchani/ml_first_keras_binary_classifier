from model import mini_conv
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from scipy import ndimage, misc
from preprocessor import train_data_generator
import sys

def predict(img_file):
    model = mini_conv()

    model.load_weights('weights/weights.h5')

    img = load_img(img_file)
    x = img_to_array(img)
    x = misc.imresize(x, (150, 150, 3))
    x = x.reshape((1,) + x.shape)
    p =  model.predict(x)

    # get label indices
    train_generator = train_data_generator()
    class_dictionary = train_generator.class_indices
    indexes = list(class_dictionary.values())
    return list(class_dictionary.keys())[indexes.index(int(p[0][0]))]


if __name__ == '__main__':
    print(predict(sys.argv[1]))