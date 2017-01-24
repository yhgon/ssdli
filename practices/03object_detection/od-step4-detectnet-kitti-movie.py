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
MODEL_JOB_DIR='/jobs/20160905-143028-2f08'
# Set the data job directory from DIGITS here
DATA_JOB_DIR='/jobs/20160905-135347-01d5'

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

BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = classifier.blobs['data'].data[...].shape

print 'The input size for the network is: (' + \
        str(BATCH_SIZE), str(CHANNELS), str(HEIGHT), str(WIDTH) + \
         ') (batch size, channels, height, width)'

# Create opencv video object
vid = cv2.VideoCapture('/data/melbourne.mp4')

# We will just use every n-th frame from the video
every_nth = 10
counter = 0

try:
    while(True):
        # Capture video frame-by-frame
        ret, frame = vid.read()
        counter += 1
        
        if not ret:
            
            # Release the Video Device if ret is false
            vid.release()
            # Mesddage to be displayed after releasing the device
            print "Released Video Resource"
            break
        if counter%every_nth == 0:
            
            # Resize the captured frame to match the DetectNet model
            frame = cv2.resize(frame, (1024, 512), 0, 0)
            
            # Use the Caffe transformer to preprocess the frame
            data = transformer.preprocess('data', frame.astype('float16')/255)
            
            # Set the preprocessed frame to be the Caffe model's data layer
            classifier.blobs['data'].data[...] = data
            
            # Measure inference time for the feed-forward operation
            start = time.time()
            # The output of DetectNet is an array of bounding box predictions
            bounding_boxes = classifier.forward()['bbox-list'][0]
            end = (time.time() - start)*1000
            
            # Convert the image from OpenCV BGR format to matplotlib RGB format for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create a copy of the image for drawing bounding boxes
            overlay = frame.copy()
            
            # Loop over the bounding box predictions and draw a rectangle for each bounding box
            for bbox in bounding_boxes:
                if  bbox.sum() > 0:
                    cv2.rectangle(overlay, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255, 0, 0), -1)
                    
            # Overlay the bounding box image on the original image
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # Display the inference time per frame
            cv2.putText(frame,"Inference time: %dms per frame" % end,
                        (10,500), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            # Display the frame
            imshow(frame)
            show()
            # Display the frame until new frame is available
            clear_output(wait=True)
            
# At any point you can stop the video playback and inference by  
# clicking on the stop (black square) icon at the top of the notebook
except KeyboardInterrupt:
    # Release the Video Device
    vid.release()
    # Message to be displayed after releasing the device
    print "Released Video Resource"
