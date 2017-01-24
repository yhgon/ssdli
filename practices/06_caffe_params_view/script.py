import sys
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a  color heatmap

import caffe

param_root = './'
caffe_root ='/home/ubuntu/caffe/'
import os
if os.path.isfile(param_root + 'bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'Need to download pre-trained CaffeNet model...'



#############################################################################################

!wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
!wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_caffenet/deploy.prototxt
!wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_caffenet/deploy.prototxt

caffe.set_mode_gpu() 


model_def = param_root + 'deploy.prototxt.1'
model_weights = param_root + 'bvlc_reference_caffenet.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(param_root + 'ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227





!ls /home/ubuntu/caffe/examples/images/


image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)



################################################################################################

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()


################################################################################################

# load ImageNet labels
labels_file = param_root + 'synset_words.txt'
if not os.path.exists(labels_file):
    !/home/ubuntu/caffe/data/ilsvrc12/get_ilsvrc_aux.sh
    
labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'output label:', labels[output_prob.argmax()]


################################################################################################

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print 'probabilities and labels:'
zip(output_prob[top_inds], labels[top_inds])


################################################################################################

# for each layer, show the output shape  -- batch size -- channel x memeory size
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)


################################################################################################
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

################################################################################################
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data) ; plt.axis('off')
    plt.show()
################################################################################################

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
################################################################################################

filters = net.params['conv1'][0].data ; vis_square(filters.transpose(0, 2, 3, 1))
filters = net.params['conv2'][0].data ; vis_square(filters[:48].reshape(48**2, 5, 5))
filters = net.params['conv3'][0].data ; vis_square(filters[:256].reshape(256**2, 3, 3))
filters = net.params['conv4'][0].data ;vis_square(filters[:192].reshape(192**2, 3, 3))
filters = net.params['conv5'][0].data ; vis_square(filters[:192].reshape(192**2, 3, 3))
################################################################################################

feat = net.blobs['conv1'].data[0, :36] ;vis_square(feat, padval=1)
feat = net.blobs['conv2'].data[0, :36] ;vis_square(feat, padval=1)
feat = net.blobs['conv3'].data[0] ;vis_square(feat, padval=0.5)
feat = net.blobs['conv4'].data[0] ;vis_square(feat, padval=0.5)
feat = net.blobs['pool5'].data[0] vis_square(feat, padval=1)
################################################################################################
