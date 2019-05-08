#!/usr/bin/env python3

# Clean system rewrite started 13:49 Friday the 1st of september
# Argument terminal fork with dynamic setting definitions
# Created by Joshua Davies

import numpy as np, os, math, sys

# ─────────────────────────────────────────────────────────────────────┤
# Options START
# ─────────────────────────────────────────────────────────────────────┤

arguments = sys.argv[1:]

resource_directory = os.path.join(os.path.dirname(__file__), 'resources')

if '-h' in arguments:
    with open(os.path.join(resource_directory, 'help.txt'), 'r') as f:
        print (f.read())
    sys.exit()

# ─────────────────────────────────────────────────────────────────────┤
# All output override:
# ─────────────────────────────────────────────────────────────────────┤
# Overrides options and outputs for some options 
# Produces outputs for if these settings had been set to true or false
# Produces for all combinations of these settings
# Overrides periodicity, flip_read and complement_strand
# True / False value activate with -ao argument
# Note: Can be incredibly slow
# ─────────────────────────────────────────────────────────────────────┤

all_output_override = '-ao' in arguments

# ─────────────────────────────────────────────────────────────────────┤
# Complement strand:
# ─────────────────────────────────────────────────────────────────────┤
# Convert input data strand to complementary strand before processing
# True / False value activate with -c argument
# ─────────────────────────────────────────────────────────────────────┤

complement_strand = '-c' in arguments

# ─────────────────────────────────────────────────────────────────────┤
# Flip read:
# ─────────────────────────────────────────────────────────────────────┤
# Reverses order of nucleotide sequence before processing
# True / False value activate with -r argument
# ─────────────────────────────────────────────────────────────────────┤

flip_read = '-r' in arguments

# ─────────────────────────────────────────────────────────────────────┤
# Input location:
# ─────────────────────────────────────────────────────────────────────┤
# Defines the path to the folder containing input data files
# Define with -f followed by a path to the input folder
# ─────────────────────────────────────────────────────────────────────┤

try:
    input_location = arguments[arguments.index('-f')+1]
except ValueError:
    input_location = os.path.join(resource_directory, 'data_files') # Change default

# ─────────────────────────────────────────────────────────────────────┤
# Output location:
# ─────────────────────────────────────────────────────────────────────┤
# Defines the path to the folder where data files are outputted
# Define with -o followed by a path to the output folder
# ─────────────────────────────────────────────────────────────────────┤

try:
    output_location = arguments[arguments.index('-o')+1]
except ValueError:
    output_location = os.path.join(resource_directory, 'output') # Change default

# ─────────────────────────────────────────────────────────────────────┤
# Output type:
# ─────────────────────────────────────────────────────────────────────┤
# Defines what kind of scale files are outputted
# averaged - Output data files for averages across all inputted scales
# individual - Output data files for each individual inputted scale
# averaged, individual - Output data files for both of the above cases
# Define with -t followed by 'a', 'i', or 'ai' 
# ─────────────────────────────────────────────────────────────────────┤

try:
    output_type = tuple(arguments[arguments.index('-t')+1])
except ValueError:
    output_type = 'a', 'i' # Change default

# ─────────────────────────────────────────────────────────────────────┤
# Periodicity assumption:
# ─────────────────────────────────────────────────────────────────────┤
# Assumes sequence repeats and wraps averaging windows around the data
# True / False value active with -p argument
# ─────────────────────────────────────────────────────────────────────┤

periodicity_assumption = '-p' in arguments

# ─────────────────────────────────────────────────────────────────────┤
# Scale location:
# ─────────────────────────────────────────────────────────────────────┤
# Defines the path to the folder containing input scale files
# Define with -s followed by a path to the input folder
# ─────────────────────────────────────────────────────────────────────┤

try:
    scale_location = arguments[arguments.index('-s')+1]
except:
    scale_location = os.path.join(resource_directory, 'scales') # Change default

# ─────────────────────────────────────────────────────────────────────┤
# Step size:
# ─────────────────────────────────────────────────────────────────────┤
# Defines the steps between window sizes if window averaging is enabled
# Integer numerical value, should be set to an even value
# Define with -ss followed by the step size as an integer value
# ─────────────────────────────────────────────────────────────────────┤

try:
    step_size = int(arguments[arguments.index('-ss')+1])
except:
    step_size = 2 # Change default

# ─────────────────────────────────────────────────────────────────────┤
# Valid extensions:
# ─────────────────────────────────────────────────────────────────────┤
# Defines file extensions the program recognises as data
# All other file types will be ignored and not listed
# Seperate extensions with a comma
# Only changeable manually due to no need for this value to be dynamic
# ─────────────────────────────────────────────────────────────────────┤

valid_extensions = '.txt', '.dat'

# ─────────────────────────────────────────────────────────────────────┤
# Window averaging:
# ─────────────────────────────────────────────────────────────────────┤
# Produces outputs from window sizes from the defined down to 3
# Useful for seeing the effect window size has on the outputted data
# Will also produce an output file with averages across all the sizes
# True / False value activate with -wa argument
# ─────────────────────────────────────────────────────────────────────┤

window_averaging = '-wa' in arguments

# ─────────────────────────────────────────────────────────────────────┤
# Window size:
# ─────────────────────────────────────────────────────────────────────┤
# Defines the width of the averaging window used when averaging data
# Also serves to define the maximum size when window averaging is on
# Integer numerical value, should be set to an odd value
# Define with -ws followed by the window size as an integer value
# ─────────────────────────────────────────────────────────────────────┤

try:
    window_size = int(arguments[arguments.index('-ws')+1])
except:
    window_size = 7 # Change default

# ─────────────────────────────────────────────────────────────────────┤
# Options END
# ─────────────────────────────────────────────────────────────────────┤

# ─────────────────────────────────────────────────────────────────────┤
# Anonymous functions START
# ─────────────────────────────────────────────────────────────────────┤

clear = lambda : os.system('clear')
divider = lambda length: '\n'+'─'*length+'\n'
files_per = lambda scales: {True:len(range(window_size, 1, -(step_size)))+1, False:1}[window_averaging] * len(scales)
setting_display = lambda: '\n'.join([var+': '+str(state) for var, state in (('Periodicity',periodicity_assumption),('Flip',flip_read),('Complement',complement_strand))])

# ─────────────────────────────────────────────────────────────────────┤
# Anonymous functions END
# ─────────────────────────────────────────────────────────────────────┤

# ─────────────────────────────────────────────────────────────────────┤
# Class definintions START
# ─────────────────────────────────────────────────────────────────────┤

class DAT_PIECE():

    def __init__(self, title, raw_array):

        self.title = title
        self.original_raw = np.array(list(''.join(raw_array)))

    def create_complement(self):

        complementary_bases = {'A':'T','T':'A','U':'A','G':'C','C':'G'}
        self.complement_raw = np.array([complementary_bases[original_base] for original_base in self.original_raw])

    def convert(self, scale):

        self.original_convert = np.array([scale[base] for base in self.original_raw])
        try:
            self.complement_convert = np.array([scale[base] for base in self.complement_raw])
        except AttributeError:
            pass

    def window_average(self, width, periodicity, reverse, complement):

        N = len(self.original_raw)
        half_width = width // 2

        if complement:
            working_array = self.complement_convert
        else:
            working_array = self.original_convert
        if reverse:
            working_array = working_array[::-1]
        if periodicity:
            sample_range = range(N)
        else:
            sample_range = range(half_width, N-half_width)

        output_array = np.array([np.mean(np.take(working_array, range(i-half_width, i+1+half_width), mode = 'wrap')) for i in sample_range])

        if not(periodicity):
            return np.concatenate((np.zeros(half_width), output_array, np.zeros(half_width)))
        return output_array

class File():

    def __init__(self, path):

        self.file = path
        self.file_name = os.path.basename(path)
        with open(self.file, 'r') as f:
            self.raw_lines = f.read().splitlines()

    def parse_lines(self, current_contents):

        for i, line in enumerate(current_contents[1:]):
            i += 1
            if line.startswith('>'):
                return [DAT_PIECE(current_contents[0][1:], current_contents[1:i])] + self.parse_lines(current_contents[i:])
        return [DAT_PIECE(current_contents[0][1:], current_contents[1:])]

    def parse_scales(self, current_contents):

        for i, line in enumerate(current_contents[1:]):
            i += 1
            if line.startswith('>'):
                new_scale = {current_contents[0][1:]:dict(zip([i.split()[0] for i in current_contents[1:i]], [float(i.split()[1]) for i in current_contents[1:i]]))}
                new_scale.update(self.parse_scales(current_contents[i:]))
                return new_scale
        new_scale = {current_contents[0][1:]:dict(zip([i.split()[0] for i in current_contents[1:]], [float(i.split()[1]) for i in current_contents[1:]]))}
        return new_scale

    def output(self, data_array):

        self.progress_bar = DualProgressBar(len(data_array), files_per(scale_data), 55, ['File '+self.file_name+' & data piece '+i.title for i in data_array])
        for data_piece in data_array:
            if complement_strand:
                data_piece.create_complement()
            file_name = ''.join([character for character, state in (('P',periodicity_assumption), ('F',flip_read), ('C',complement_strand)) if state == True])
            if file_name == '':
                file_name = 'vanilla'
            for scale_title in scale_data:
                scale = scale_data[scale_title]
                data_piece.convert(scale)
                output_arrays = [data_piece.window_average(i, periodicity_assumption, flip_read, complement_strand) for i in range(window_size, 2, -(step_size))]
                file_path = os.path.join(output_location, data_piece.title, scale_title)
                self.write_file(output_arrays[0], file_name, file_path, window_size, periodicity_assumption)
                if window_averaging:
                    for i, array in enumerate(output_arrays[1:]):
                        width = window_size - ((i+1) * step_size)
                        self.write_file(array, file_name, file_path, width, periodicity_assumption)
                    self.write_file(average_arrays(output_arrays), file_name, file_path, 'avg', periodicity_assumption)

    def write_file(self, output_array, file_name, file_path, width, periodicity):

        self.progress_bar.step()
        file_name = file_name+'W'+str(width)+'.txt'
        file_path = os.path.join(file_path, file_name)
        if width != 'avg':
            lyapunov = return_lyapunov(output_array, periodicity, width)
        else:
            lyapunov = 'unavailable for averaged data sets'
        if not(os.path.exists(os.path.dirname(file_path))):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            output_lines = ['# lyapunov = '+str(lyapunov)+'\n']+[str(x) + ' ' + str(y) + '\n' for x, y in enumerate(output_array)]
            f.writelines(output_lines)

class DualProgressBar():

    def __init__(self, master_max, sub_max, length, headers):

        self.__dict__.update(locals())
        self.sub_progress = 0
        self.master_progress = 0

    def step(self):

        self.sub_progress += 1
        self.master_progress += self.sub_progress // self.sub_max
        if self.master_progress != self.master_max:
            self.sub_progress %= self.sub_max
        self.update()

    def update(self):

        clear()
        try:
            print ('Processing '+self.headers[self.master_progress])
        except IndexError:
            print ('Done!')
        print (divider(55)+'Current settings: '+divider(55)+setting_display()+divider(55))
        for progress, maximum in (self.sub_progress, self.sub_max), (self.master_progress, self.master_max):
            completed = int((progress / maximum) * self.length)
            remaining = self.length - completed
            print ('\n'+str(progress)+'/'+str(maximum)+divider(55)+'▮'*completed+'-'*remaining+divider(55))


# ─────────────────────────────────────────────────────────────────────┤
# Class definitions END
# ─────────────────────────────────────────────────────────────────────┤

# ─────────────────────────────────────────────────────────────────────┤
# Function definitions BEGIN
# ─────────────────────────────────────────────────────────────────────┤

def average_arrays(array_tuple):

    zipped = list(zip(*array_tuple))
    output = [np.mean(i) for i in zipped]
    return (output)

def average_dictionaries(dictionary_tuple):

    output = {key:np.mean([dictionary[key] for dictionary in dictionary_tuple]) for key in dictionary_tuple[0]}
    return output

def merge_dictionaries(dictionary_tuple):

    [dictionary_tuple[0].update(i) for i in dictionary_tuple[1:]]
    return dictionary_tuple[0]

def nearest_neighbour(array, item, mean_period):

    for i in np.delete(array, item):
        distance = abs(item-i)
        try:
            if distance < shortest_distance and distance > mean_period:
                shortest_distance = distance
        except UnboundLocalError:
            if distance > mean_period:
                shortest_distance = distance
    return shortest_distance

def return_lyapunov(array, periodicity, width):

    half_size = width // 2
    if not(periodicity):
        array = array[half_size+1:-half_size]
    N = len(array)
    unique_counts = dict(zip(*np.unique(array, return_counts = True)))
    mean_frequency = np.sum([unique_counts[key] / N for key in array])
    mean_period = 1 / mean_frequency
    differences = np.array([nearest_neighbour(array, i, mean_period) for i in range(N)])
    lambdas = np.array([(1/ (N-i))*math.log(abs(np.take(differences, (i+1), mode = 'wrap') / np.take(differences, (i), mode = 'wrap'))) for i in range(N)])
    lyapunov = np.sum([lambdas[i] / (N-1) for i in range(N)])
    return lyapunov

def option_menu(titles, container, prompt):

    print ('\na Use all files')
    print (''.join(['\n'+str(i)+' '+title+'\n' for i, title in enumerate(titles)]))
    while True:
        try:
            user_selection = input(prompt).split()
            if user_selection == []:
                raise ValueError
            if 'a' in user_selection:
                return container
            if all([int(i) in range(len(container)) for i in user_selection]):
                return np.take(container, user_selection)
        except ValueError:
            print ('\nInvalid entry! ensure your entrys are present in the menu')

# ─────────────────────────────────────────────────────────────────────┤
# Function definitions END
# ─────────────────────────────────────────────────────────────────────┤

# ─────────────────────────────────────────────────────────────────────┤
# Scale loading code BEGIN
# ─────────────────────────────────────────────────────────────────────┤

scale_files = [File(os.path.join(scale_location, scale_file)) for scale_file in os.listdir(scale_location)]
scale_data = merge_dictionaries([scale_file.parse_scales(scale_file.raw_lines) for scale_file in scale_files])
if 'a' in output_type and not('i' in output_type):
    scale_data = {'Averaged':average_dictionaries(tuple(scale_data.values()))}
elif 'a' in output_type:
    scale_data['Averaged'] = average_dictionaries(tuple(scale_data.values()))

# ─────────────────────────────────────────────────────────────────────┤
# Scale loading code END
# ─────────────────────────────────────────────────────────────────────┤

# ─────────────────────────────────────────────────────────────────────┤
# File loading code BEGIN
# ─────────────────────────────────────────────────────────────────────┤

data_files = [File(os.path.join(input_location, data_file)) for data_file in os.listdir(input_location)]
data_files = option_menu([data_file.file_name for data_file in data_files], data_files, 'Multiple data files detected, enter one or more options seperated by a space: ')
if not(all_output_override):
    [data_file.output(data_file.parse_lines(data_file.raw_lines))for data_file in data_files]
else:
    window_averaging = True
    for periodicity_assumption in (True, False):
        for flip_read in (True, False):
            for complement_strand in (True, False):
                [data_file.output(data_file.parse_lines(data_file.raw_lines))for data_file in data_files]

# ─────────────────────────────────────────────────────────────────────┤
# File loading code END
# ─────────────────────────────────────────────────────────────────────┤
