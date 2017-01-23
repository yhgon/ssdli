import os
import aifc
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from skimage import io
from scipy.misc import imsave

def aiff2amplitudes(aiff_path):
	s = aifc.open(aiff_path, 'r')
	nframes = s.getnframes() #The total number of audio frames in the file
	strsig = s.readframes(nframes) #Returned data is a string containing for each frame the uncompressed samples of all channels
	return np.fromstring(strsig, np.short).byteswap() 
	
def amplitudes2spectrogram(amplitudes):
	return_data = plt.specgram(amplitudes,NFFT=256,noverlap=128)
        pxx = return_data[0]
	return pxx

def convert(image_folder):
	for root, dirs, filenames in os.walk(image_folder):
    		for f in filenames:
			if os.path.splitext(f)[-1] == '.aiff':
				amplitudes = aiff2amplitudes(os.path.join(root,f))
				spectrogram = amplitudes2spectrogram(amplitudes)
				out_name = os.path.splitext(f)[-2] + '.png'
				imsave(os.path.join(root,out_name),spectrogram)
				print out_name

if __name__=="__main__":
	
	# Convert training data
	convert('data/train')

	# Convert testing data
	#convert('data/test')
