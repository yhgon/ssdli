# Import required Python libraries
%pylab inline
pylab.rcParams['figure.figsize'] = (15, 9)
import caffe
import numpy as np
import time
import os
import cv2
from IPython.display import clear_output

# Configure Caffe to use the GPU for inference
caffe.set_mode_gpu()

# Set the model job directory from DIGITS here
MODEL_JOB_DIR='/home/ubuntu/digits/digits/jobs/20160905-143028-2f08'
# Set the data job directory from DIGITS here
DATA_JOB_DIR='/home/ubuntu/digits/digits/jobs/20160905-135347-01d5'

# We need to find the iteration number of the final model snapshot saved by DIGITS
for root, dirs, files in os.walk(MODEL_JOB_DIR):
    for f in files:
        if f.endswith('.solverstate'):
            last_iteration = f.split('_')[2].split('.')[0]
print 'Last snapshot was after iteration: ' + last_iteration

# Load the dataset mean image file
mean = np.load(os.path.join(DATA_JOB_DIR,'train_db','mean.npy'))

# Instantiate a Caffe model in GPU memory
# The model architecture is defined in the deploy.prototxt file
# The pretrained model weights are contained in the snapshot_iter_<number>.caffemodel file
classifier = caffe.Net(os.path.join(MODEL_JOB_DIR,'deploy.prototxt'), 
                       os.path.join(MODEL_JOB_DIR,'snapshot_iter_' + last_iteration + '.caffemodel'),
                       caffe.TEST)

# Instantiate a Caffe Transformer object that wil preprocess test images before inference
transformer = caffe.io.Transformer({'data': classifier.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data',mean.mean(1).mean(1)/255)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

NEW_BATCH_SIZE = 2

# Resize the input data layer for the new batch size
OLD_BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = classifier.blobs['data'].data[...].shape
classifier.blobs['data'].reshape(NEW_BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
classifier.reshape()

# Create opencv video object
vid = cv2.VideoCapture('/home/ubuntu/deployment_lab/melbourne.mp4')

counter = 0

batch = np.zeros((NEW_BATCH_SIZE, HEIGHT, WIDTH, CHANNELS))

try:
    while(True):
        # Capture video frame-by-frame
        ret, frame = vid.read()
        
        if not ret:
            # Release the Video Device if ret is false
            vid.release()
            # Mesddage to be displayed after releasing the device
            print "Released Video Resource"
            break
            
        # Resize the captured frame to match the DetectNet model
        frame = cv2.resize(frame, (WIDTH, HEIGHT), 0, 0)
        
        # Add frame to batch array
        batch[counter%NEW_BATCH_SIZE,:,:,:] = frame
        counter += 1
        
        if counter%NEW_BATCH_SIZE==0:
        
            # Use the Caffe transformer to preprocess the frame
            data = transformer.preprocess('data', frame.astype('float16')/255)
            
            # Set the preprocessed frame to be the Caffe model's data layer
            classifier.blobs['data'].data[...] = data
            
            # Measure inference time for the feed-forward operation
            start = time.time()
            # The output of DetectNet is now an array of bounding box predictions
            # for each image in the input batch
            bounding_boxes = classifier.forward()['bbox-list']
            end = (time.time() - start)*1000
            
            print 'Inference time: %dms per batch, %dms per frame, output size %s' % \
                    (end, end/NEW_BATCH_SIZE, bounding_boxes.shape)
            
# At any point you can stop the video playback and inference by  
# clicking on the stop (black square) icon at the top of the notebook
except KeyboardInterrupt:
    # Release the Video Device
    vid.release()
    # Message to be displayed after releasing the device
    print "Released Video Resource"
