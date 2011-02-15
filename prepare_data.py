import os
import glob
import scipy as sp
import Image


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

	# -- get all images
	images = []
	for item in class_list:
		#get list
		im_list = glob.glob(os.path.join(item, '*.jpg'))
		im_list = sp.random.permutation(im_list)
		
		#get images as arrays
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

	train_data = [''] * (train_n * class_n)
	valid_data = [''] * (valid_n * class_n)
	test_data = [''] * (test_n * class_n)

	train_labels = sp.zeros((train_n * class_n, 1), dtype = int)
	valid_labels = sp.zeros((valid_n * class_n, 1), dtype = int)
	test_labels = sp.zeros((test_n * class_n, 1), dtype = int)
	

	# -- loop over all classes to divide data
	for i_ind in range(class_n):
		# -- divide data into train, validation and test
		train_rng = xrange(i_ind * train_n, (i_ind + 1) * train_n)
		valid_rng = xrange(i_ind * valid_n, (i_ind + 1) * valid_n)
		test_rng = xrange(i_ind * test_n, (i_ind + 1) * test_n)

		# data
		train_data[train_rng] = images[i_ind][:train_n]
		valid_data[valid_rng] = images[i_ind][train_n: train_n + valid_n]
		test_data[test_rng] = images[i_ind][train_n + valid_n:]

		# labels
		train_lables[train_rng] = i_ind
		valid_lables[valid_rng] = i_ind
		test_lables[test_rn] = i_ind

	# -- final shuffle of data
	shuffle_ind = sp.random.permutation(train_n * n_class)
	train_data = train_data[shuffle_ind]
	train_labels = train_labels[shuffle_ind]

	shuffle_ind = sp.random.permutation(valid_n * n_class)
	valid_data = valid_data[shuffle_ind]
	train_labels = valid_labels[shuffle_ind]
	
	shuffle_ind = sp.random.permutation(test_n * n_class)
	test_data = test_data[shuffle_ind]
	test_labels = test_labels[shuffle_ind]

	return (train_data, train_labels), (valid_data, valid_labels),\
			(test_data, test_labels)

	

if __name__ == "__main__":
	global verbose
	verbose = False


	# -- get images
	img_path = '/idiap/home/mmirza/data/image_net/'
	images, class_names, im_n_limit = get_image_list(img_path)
	divide_data(images, im_n_limit, 50, 10)
	
