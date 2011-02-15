""" A module for preparing data for object categorization task

	Read images from diffrent classes each in a seprate folder,
	and gray scale, rezie images and save them as pickle files in 
	multiple chunks

mehdi.mirza@idiap.ch
February 2011
"""



import os
import glob
import scipy as sp
import Image
import pickle
import argparse

def get_image(img_fname, new_size = None, resize_method='bicubic'):
	""" Return a grayscaled and resized image as numpy array

	Inputs:
		img_fname: Image file name
		new_size: new image size, if None just check image and
					return image file name
		resize_method: 'antialias' or 'bicubic'

	Outputs:
		imga: resutls
	"""

	# -- open image and convert to grayscale
	try:
		img = Image.open(img_fname).convert('L')
	except IOError:
		if verbose:
			print "Failed opening: ",img_fname
		return None

	if new_size != None:
		# -- resize image
		if resize_method.lower() == 'bicubic':
			img = img.resize(new_size, Image.BICUBIC)
		elif resize_method.lower() == 'antialias':
			img = img.resize(new_size, Image.ANTIALIAS)

		else:
			raise ValueError("resize_method '%s' not understood", resize_method)

		# -- convert to a numpy array
		imga = sp.misc.fromimage(img)/255.
	else:
		return img_fname

	return imga


def get_image_list(img_path, im_size = None, class_list = None):
	""" Return list all images and class names
	
	Inputs:
		img_path: Path to image folders (If is empty pwd will be used)
		im_size: Size to which change images to. If None just return image 
					file names
		class_list: List of classes (If None, all folders in img_path will be
					considerd)

	Outputs:
		images: list of all images, separated in diffrent list for each class
		class_list: list of path of classes
		min_n: minimum nuber of images in a class
	"""
	
	# -- check path
	if img_path == '':
		img_path = os.path.cudir()
	assert os.path.isdir(img_path)

	# -- get list of classes
	if class_list == None:
		class_list = [ os.path.join(img_path, name) for name in \
					os.listdir(img_path) if\
					os.path.isdir(os.path.join(img_path, name)) ]
	else:
		class_list= [os.path.join(img_path, name) for name in \
					class_list]
		# -- ToDo assert if list values are valid

	# -- get list of images
	images = []
	for item in class_list:
		#get list
		im_list = glob.glob(os.path.join(item, '*.jpg'))
		im_list = sp.random.permutation(im_list)
		
		#get images as array or just check if the image is openable
		#if im_size is none and return the file name
		data = [get_image(name, im_size) for name in im_list]

		#remove None values
		for i_ind in range (data.count(None)):
			data.remove(None)
		
		images.append(data)
		
	# -- minmium number of images in all classes
	min_n = min([len(item) for item in images])

	return images, class_list, min_n


def divide_data(images, im_n_limit, train_r, valid_r):
	"""Divide data to train, validation and test subsets

	Inputs:
		images: a list of all array images
		im_n_limit: image number limit for all classes
		train_r: % of train data
		valid_r: % of validation data
				(Remaning will be used as testing data)
					considerd)

	Outputs:
		train: a tuple of traind data and their label
		validation: same as above for validation data
		test: same as above for test data
	"""


	train_n = int(im_n_limit * train_r / 100)
	valid_n = int(im_n_limit * valid_r / 100)
	test_n = im_n_limit - (train_n + valid_n)


	# -- intialize data
	class_n = len(images)

	train_data = []
	valid_data = []
	test_data = []

	train_labels = []
	valid_labels = []
	test_labels = []
	

	# -- loop over all classes to divide data
	for i_ind in range(class_n):
		# -- divide data into train, validation and test

		# data
		train_data.extend(images[i_ind][:train_n])
		valid_data.extend(images[i_ind][train_n: train_n + valid_n])
		test_data.extend(images[i_ind][train_n + valid_n:im_n_limit])

		# labels
		train_labels.extend([i_ind] * train_n)
		valid_labels.extend([i_ind] * valid_n)
		test_labels.extend([i_ind] * test_n)

	# -- final shuffle of data
	shuffle_ind = sp.random.permutation(train_n * class_n)
	train_data = sp.array(train_data)[shuffle_ind]
	train_labels = sp.array(train_labels)[shuffle_ind]

	shuffle_ind = sp.random.permutation(valid_n * class_n)
	valid_data = sp.array(valid_data)[shuffle_ind]
	valid_labels = sp.array(valid_labels)[shuffle_ind]
	
	shuffle_ind = sp.random.permutation(test_n * class_n)
	test_data = sp.array(test_data)[shuffle_ind]
	test_labels = sp.array(test_labels)[shuffle_ind]


	return (train_data, train_labels), (valid_data, valid_labels),\
			(test_data, test_labels)

	

def chunk_data(data, chunk_size, im_size, save_path, name_pattern):
	"""Chunk the data and load images and save images data and
		labels for each chunk
	
	Inputs
		data: a tuple of arrays of image file names and labels
		chunk_size: size of each chink
		im_size: image size
		save_path: output files path
		name_pattern: pattern for naming output files

	"""

	# -- preapre output path
	if not os.path.isdir(save_path):
		raise NameError('Invalid path: %s'% save_path)

	# -- prepare data
	data, labels = data
	chunk_num = data.shape[0] / chunk_size

	# -- loop over chunks, load data, pickle them
	for i_ind in range(chunk_num):
		images = data[i_ind*chunk_size:(i_ind+1) * chunk_size]
		# --load image data
		images = sp.array([get_image(name, im_size) for name in images])
	
		# --pickle
		fname = os.path.join(save_path, "%s_%03d.pkl" % (name_pattern, i_ind))
		output = open(fname , 'wb')
		pickle.dump(images, output, -1)
		pickle.dump(labels[i_ind*chunk_size:(i_ind) + 1 * chunk_size], 
				output, -1)
		output.close()

		if verbose:
			print "Pickled %s - %d/%d" % (fname, i_ind, chunk_num)

def chunk_size(im_size, memory_size, data_type):
	"""Calculate number of images in each chunk that could fit in memory

	Inputs:
		im_size: image size tuple
		memory_size: memory size limit in MB
		data_type: numpy data type that is used for saving image arrays

	Output:
		chunk_size: chunk size interger
	"""
	
	# -- convert Mb to bytes
	memory_size = 1048576 * memory_size
	data_type_size = sp.array([0], dtype = data_type).itemsize
	chunk_size =  memory_size / (im_size[0] * im_size[1] * \
					data_type_size)

	return chunk_size

def main():

	# -- parse argumnets
	parser = argparse.ArgumentParser(description='Preapare image data in \
		pickle files, divided in train, valid and test subset')
	parser.add_argument('-i', '--input', dest = 'input', help =\
		'path to folder that images are located', required = True)
	parser.add_argument('-o', '--output', dest = 'output', help =\
		'output path for pickle files', required = True)
	parser.add_argument('-s', '--size', dest = 'im_size',
		help = 'Size to change image to. One number reqired, \
		will make images square', required = True, type = int)
	parser.add_argument('-m', '--memsize', dest = 'mem_size',
		help = 'GPU global memory size in Mb', default = 30, type = int)
	parser.add_argument('-t', '--type', dest='data_type' , choices = [32, 64],
		help = 'image array data types, 32: for floate32 \
		64: for float64. (Many GPU\'s just support float32')
	parser.add_argument('--train', dest = 'train', type = int, 
		default = 50, help = '% of data for training')
	parser.add_argument('--valid', dest = 'valid', type = int, 
		default = 10, help = '% of data for training. \
		Note: the rest would be for test')
	parser.add_argument('-n', '--name', dest='name', default='data',
		help = 'name pattern for output files')
	parser.add_argument('-v', '--verbose', default = False,
		action = 'store_true')
	
	args = parser.parse_args()

	# -- initialize variables
	global verbose
	verbose = args.verbose

	data_type = sp.float32
	if args.data_type == 64:
		data_type = sp.float64

	# -- main

	#get list of files
	images, class_names, im_n_limit = get_image_list(args.input)
	#dvidie data
	train, valid, test = divide_data(images, im_n_limit, 
							args.train, args.valid)
	#chunk and save it
	chunk =  chunk_size((args.im_size, args.im_size), args.mem_size, 
						data_type)
	chunk_data(train, chunk, (args.im_size, args.im_size),
				args.output, args.name + '_train')
	chunk_data(valid, chunk, (args.im_size, args.im_size),
				args.output, args.name + '_valid' )
	chunk_data(test, chunk, (args.im_size, args.im_size),
				args.output, args.name + '_test')



if __name__ == "__main__":

	main()

