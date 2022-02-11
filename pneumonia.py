
from PIL import Image
from numpy import asarray
import numpy as np
import os


class Preprocess:
  def __init__(self,directory=None):
    self.base_directory = directory

    #the below function will return the tuple of list of matrix(desired_row X desired_col) of each 
    #image present in the given directory along with the seperate list of targets(mentioned in the 
    #initialisation). if get_as_vector = True then one of the returning part tuple is a list of vectors
    #instead of matrix
  def get_all_photos_as_array(self,target=0,folder_name=None,desired_row=100,desired_col=100,dec_factor=255,get_as_vector=False):
    _base_directory = self.base_directory + '/' + folder_name
    directory = os.fsencode(_base_directory)
    file_array = []
    #The below loop iterates over every file in the given directory
    for file in os.listdir(directory):
      filename = os.fsdecode(file)
      #if there is .jpeg or .png extension in the file that will file path will be added to file_array
      if filename.endswith(".jpeg") or filename.endswith(".png"): 
         temp_direc = _base_directory + '/' + filename
         file_array.append(temp_direc)
         continue
      else:
         continue

    n_rows = len(file_array) 
    desired_size = (desired_row,desired_col)
    #if user choosed to get output as list of vectors then shape of data_input_array will be changed accordingly
    if get_as_vector:
      data_input_array = np.ndarray(shape = (n_rows,desired_size[0]*desired_size[1]))
      data_input_array = data_input_array.astype('float64')
    else:
      data_input_array = np.ndarray(shape = (n_rows,desired_size[0],desired_size[1]))
      data_input_array = data_input_array.astype('float64')

    #we will be iterating through the file array that we got above and change every image in it into matrix
    #and store in data_input_array (the shape is vector if it get_as_vector = True else it will be tensor)
    for single_file in file_array:
      image = Image.open(single_file)
      image_data = asarray(image)
      image_data = image_data.astype('float64')
      image_data = np.resize(image_data,desired_size)
      image_data = image_data/dec_factor
      if get_as_vector:
        total_len = image_data.shape[0] * image_data.shape[1]
        image_data = image_data.reshape(total_len,)
      data_input_array = np.concatenate((data_input_array, [image_data]), axis=0)

    #These will be target array that we will be difining 
    #user gets to choose it at initialization of the instance of this class
    if target == 0:
      target_array = np.full(len(data_input_array),'no')
    elif target == 1:
      target_array = np.full(len(data_input_array),'yes')
    
    return data_input_array,target_array

    #The above function is just as above but instead of iterating over all the files in the mentioned folder
    #it will be return the array of the given single image
  def get_photo_array(self,file_name=None,desired_row=100,desired_col=100,dec_factor=255,get_as_vector=False):
    desired_size = (desired_row,desired_col)
    directory = self.base_directory + '/' + file_name
    image = Image.open(directory)
    image_data = asarray(image)
    image_data = image_data.astype('float64')
    image_data = np.resize(image_data,desired_size)
    image_data = image_data/dec_factor
    if get_as_vector:
      total_len = image_data.shape[0] * image_data.shape[1]
      image_data = image_data.reshape(total_len,)
    return image_data
  #The below method will take two input arrays as arguments and adds up them row wise and returns array
  def add_these_row_wise(self,inputs_1,inputs_2):
    inputs = np.concatenate((inputs_1,inputs_2), axis=0)
    return inputs
  #will take two arrays i.e inputs and targets and shuffle them in same order and return them
  def shuffle_inputs_and_targets(self,inputs,targets):
    shuffled_indices = np.arange(inputs.shape[0])
    np.random.shuffle(shuffled_indices)
    inputs = inputs[shuffled_indices]
    targets = targets[shuffled_indices]
    return inputs,targets
  
  def balance_targets(self,inputs,targets,shuffle=False):
    low_count = np.count_nonzero(targets == 'yes')
    high_count = np.count_nonzero(targets == 'no')
    high,low = 'no','yes'
    if high_count < low_count:
      high = 'yes'
      check = high_count
      high_count = low_count
      low_count = check 
      low = 'no'
    counter = 0
    remove_indexes = []
    for i in range(targets.shape[0]):
      if targets[i] == low:
        continue
      else:
        counter += 1
        if counter > low_count:
          remove_indexes.append(i)
    targets = np.delete(targets,remove_indexes,axis=0)
    inputs = np.delete(inputs,remove_indexes,axis=0)
    if shuffle:
      return self.shuffle_inputs_and_targets(inputs,targets)
    return inputs,targets
