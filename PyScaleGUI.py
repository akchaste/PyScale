# Clean system rewrite started 13:49 Friday the 1st of september
# GUI fork with dynamic setting changing
# Created by Joshua Davies

import numpy as np, os, math, tkinter.filedialog as filedialog, tkinter.messagebox as messagebox
from tkinter import *
from tkinter import ttk
from shutil import rmtree

root = Tk()
root.resizable(width = False, height = False)

plot_calls = 0

# ─────────────────────────────────────────────────────────────────────┤
# Basic GUI format START
# ─────────────────────────────────────────────────────────────────────┤

# ─────────────────────────────────────────────────────────────────────┤
# Option toolbar (TOP)
# ─────────────────────────────────────────────────────────────────────┤
# Where options are modified and program outputs are initialised
# ─────────────────────────────────────────────────────────────────────┤

option_toolbar = Frame(root, bd = 1, relief = 'solid')
option_toolbar.grid(column = 0, columnspan = 2, sticky = (N,E,W), padx = 5, pady = 5)

# ─────────────────────────────────────────────────────────────────────┤
# File browser (BOTTOM LEFT)
# ─────────────────────────────────────────────────────────────────────┤
# Displays outputted files and launches plotting window
# ─────────────────────────────────────────────────────────────────────┤

file_browser = Frame(root, bd = 1, relief = 'solid')
file_browser.grid(column = 0, row = 1, sticky = (N,S,W), padx = 5, pady = 5)

file_tree = ttk.Treeview(file_browser, height = 15)
file_tree.grid(column = 0, row = 4, padx = 5, pady = 5)

# ─────────────────────────────────────────────────────────────────────┤
# Progress updater (BOTTOM RIGHT)
# ─────────────────────────────────────────────────────────────────────┤
# Displays progress of file output to keep user updated
# ─────────────────────────────────────────────────────────────────────┤

progress_updater = Frame(root, bd = 1, relief = 'solid')
progress_updater.grid(column = 1, row = 1, sticky = (N,S,W), padx = 5, pady = 5)
progress_output = Text(progress_updater, height = 30, width = 224, state = DISABLED)
progress_output.grid(column = 0, row = 0)

# ─────────────────────────────────────────────────────────────────────┤
# Basic GUI format END
# ─────────────────────────────────────────────────────────────────────┤

# ─────────────────────────────────────────────────────────────────────┤
# Default option values START
# ─────────────────────────────────────────────────────────────────────┤

# ─────────────────────────────────────────────────────────────────────┤
# All output override:
# ─────────────────────────────────────────────────────────────────────┤
# Change within program, defines default only
# ─────────────────────────────────────────────────────────────────────┤

all_output_override = False

# ─────────────────────────────────────────────────────────────────────┤
# Complement strand:
# ─────────────────────────────────────────────────────────────────────┤
# Change within program, defines default only
# ─────────────────────────────────────────────────────────────────────┤

complement_strand = False

# ─────────────────────────────────────────────────────────────────────┤
# Flip read:
# ─────────────────────────────────────────────────────────────────────┤
# Change within program, defines default only
# ─────────────────────────────────────────────────────────────────────┤

flip_read = False

# ─────────────────────────────────────────────────────────────────────┤
# Input location:
# ─────────────────────────────────────────────────────────────────────┤
# Change within program, defines default only
# ─────────────────────────────────────────────────────────────────────┤

input_location = 'resources/data_files'

# ─────────────────────────────────────────────────────────────────────┤
# Output location:
# ─────────────────────────────────────────────────────────────────────┤
# Change within program, defines default only     
# ─────────────────────────────────────────────────────────────────────┤

output_location = 'resources/output'

# ─────────────────────────────────────────────────────────────────────┤
# Output type:
# ─────────────────────────────────────────────────────────────────────┤
# Change within program, defines default only
# ─────────────────────────────────────────────────────────────────────┤

output_type = 'ai'

# ─────────────────────────────────────────────────────────────────────┤
# Periodicity assumption:
# ─────────────────────────────────────────────────────────────────────┤
# Change within program, defines default only
# ─────────────────────────────────────────────────────────────────────┤

periodicity_assumption = False

# ─────────────────────────────────────────────────────────────────────┤
# Scale location:
# ─────────────────────────────────────────────────────────────────────┤
# Change within program, defines default only
# ─────────────────────────────────────────────────────────────────────┤

scale_location = 'resources/scales'

# ─────────────────────────────────────────────────────────────────────┤
# Step size:
# ─────────────────────────────────────────────────────────────────────┤
# Change within program, defines default only
# ─────────────────────────────────────────────────────────────────────┤

step_size = 2

# ─────────────────────────────────────────────────────────────────────┤
# Valid extensions:
# ─────────────────────────────────────────────────────────────────────┤
# Defines file extensions the program recognises as data
# All other file types will be ignored and not listed
# Seperate extensions with a comma
# ─────────────────────────────────────────────────────────────────────┤

valid_extensions = '.txt', '.dat'

# ─────────────────────────────────────────────────────────────────────┤
# Window averaging:
# ─────────────────────────────────────────────────────────────────────┤
# Change within program, defines default only
# ─────────────────────────────────────────────────────────────────────┤

window_averaging = False

# ─────────────────────────────────────────────────────────────────────┤
# Window size:
# ─────────────────────────────────────────────────────────────────────┤
# Change within program, defines default only
# ─────────────────────────────────────────────────────────────────────┤

window_size = 7

# ─────────────────────────────────────────────────────────────────────┤
# Default option values END
# ─────────────────────────────────────────────────────────────────────┤

# ─────────────────────────────────────────────────────────────────────┤
# Anonymous functions START
# ─────────────────────────────────────────────────────────────────────┤

clear = lambda : progress_output.delete(1.0, END)
divider = lambda master, length: Frame(master, height = 3, bd = 1, width = length, relief = 'ridge')
textdivider = lambda length: '\n'+'─'*length+'\n'
files_per = lambda scales: {True:len(range(window_size, 1, -(step_size)))+1, False:1}[window_averaging] * len(scales)
setting_display = lambda: '\n'.join([var+': '+str(state) for var, state in (('Periodicity',periodicity_assumption),('Flip',flip_read),('Complement',complement_strand))])

# ─────────────────────────────────────────────────────────────────────┤
# Anonymous functions END
# ─────────────────────────────────────────────────────────────────────┤

# ─────────────────────────────────────────────────────────────────────┤
# Class definintions START
# ─────────────────────────────────────────────────────────────────────┤

class Options():

    def __init__(self, **kwargs):

        for key, value in kwargs.items():
            setattr(self, key, {str:StringVar(), bool:BooleanVar(), int:IntVar()}[type(value)])
            getattr(self, key).set(value)

    def globalise(self):

        current_values = {key:self.__dict__[key].get() for key in self.__dict__}
        globals().update(current_values)

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

    def output(self, data_array, scale_data):

        self.progress_bar = DualProgressBar(len(data_array), files_per(scale_data), 160, ['File '+self.file_name+' & data piece '+i.title for i in data_array])
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

        progress_output.configure(state = 'normal')
        clear()
        try:
            progress_output.insert('end', 'Processing :'+self.headers[self.master_progress]+textdivider(160))
        except IndexError:
            progress_output.insert('end', 'Done!'+textdivider(160))
        progress_output.insert('end', 'Current settings: \n'+setting_display()+textdivider(160))
        for progress, maximum in (self.sub_progress, self.sub_max), (self.master_progress, self.master_max):
            completed = int((progress / maximum) * self.length)
            remaining = self.length - completed
            progress_output.insert('end', '\n'+str(progress).zfill(2)+'/'+str(maximum).zfill(2)+textdivider(160)+'▮'*completed+'-'*remaining+textdivider(160))
        progress_output.configure(state = 'disabled')
        root.update()
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

def purge_directory():
    confirm = messagebox.askquestion('Purge Directory', 'Are you sure you want to delete all files and folders present in the output directory?')
    if confirm == "yes":
        for i in os.listdir(output_location):
            rmtree(os.path.join(output_location, i))
        messagebox.showinfo('Cleared', 'Directory purged')
        scan_output(0, output_location)


def produce_output():

    options.globalise()

    # ───────────────────────────────────────────────────────────────--|
    # Load scale files
    # ───────────────────────────────────────────────────────────────--|

    scale_files = [File(os.path.join(scale_location, scale_file)) for scale_file in os.listdir(scale_location)]
    scale_data = merge_dictionaries([scale_file.parse_scales(scale_file.raw_lines) for scale_file in scale_files])
    if 'a' in output_type and not('i' in output_type):
        scale_data = {'Averaged':average_dictionaries(tuple(scale_data.values()))}
    elif 'a' in output_type:
        scale_data['Averaged'] = average_dictionaries(tuple(scale_data.values()))

    # ───────────────────────────────────────────────────────────────--|
    # Load data files
    # ───────────────────────────────────────────────────────────────--|

    data_files = [File(os.path.join(input_location, data_file)) for data_file in os.listdir(input_location)]
    if not(all_output_override):
        [data_file.output(data_file.parse_lines(data_file.raw_lines), scale_data)for data_file in data_files]
    else:
        options.window_averaging.set(True)
        for periodicity_assumption in (True, False):
            options.periodicity_assumption.set(periodicity_assumption)
            for flip_read in (True, False):
                options.flip_read.set(flip_read)
                for complement_strand in (True, False):
                    options.complement_strand.set(complement_strand)
                    options.globalise()
                    [data_file.output(data_file.parse_lines(data_file.raw_lines), scale_data)for data_file in data_files]

    scan_output(0, output_location)

def scan_output(depth, current_folder):

    if depth == 3:
        return
    if depth == 0:
        file_tree.delete(*file_tree.get_children())
    for i in sorted(os.listdir(current_folder)):
        if i.startswith('.'):
            continue
        new_folder = os.path.join(current_folder, i)
        parent = current_folder
        if depth == 0:
            parent = ''
        try:
            file_tree.insert(parent, index = 'end', iid = new_folder, text = i[:20]+'..')
        except:
            pass
        scan_output(depth+1, new_folder)

def plot_selected():

    global plot_calls
    selected_file = file_tree.focus()
#    format_string = ["--" , "o" , "^" , "s" , "h" , "p", "x"]
    format_string = ["--" , "-." , "-x" , "-D" , "-h" , "-p", "-o"]
    try:
        if not(selected_file.endswith('.txt')):
            raise FileNotFoundError
        data = np.loadtxt(selected_file)
        plt.plot(data[:,0], data[:,1], format_string[plot_calls%len('markers')], label = selected_file)
#       plt.legend()
        plt.xlabel('Sequence Position')
        plt.ylabel('Propensity Scale')
        plt.show(block = False)
        plot_calls += 1
    except (FileNotFoundError, NameError) as current_error:
        if type(current_error) == FileNotFoundError:
            messagebox.showerror('Error','Ensure you selected a valid data file!')
        else:
            messagebox.showerror('Error','Plotting disabled due to no matplotlib install')

# ─────────────────────────────────────────────────────────────────────┤
# Function definitions END
# ─────────────────────────────────────────────────────────────────────┤

# ─────────────────────────────────────────────────────────────────────┤
# Dynamic option initialisation START
# ─────────────────────────────────────────────────────────────────────┤

variable_names = 'all_output_override', 'complement_strand', 'flip_read', 'input_location', 'output_location', 'output_type', 'periodicity_assumption', \
                 'scale_location', 'step_size', 'window_averaging', 'window_size'

options = Options(**{name:eval(name) for name in variable_names})
variable_zip = zip(variable_names, [i.replace('_', ' ').capitalize() for i in variable_names])

# ─────────────────────────────────────────────────────────────────────┤
# Dynamic option initialisation END
# ─────────────────────────────────────────────────────────────────────┤

# ─────────────────────────────────────────────────────────────────────┤
# Toolbar population START
# ─────────────────────────────────────────────────────────────────────┤

# ─────────────────────────────────────────────────────────────────────┤
# Create True/False value checkbuttons
# ─────────────────────────────────────────────────────────────────────┤

Label(option_toolbar, text = 'Generic options:').grid(column = 0, columnspan = 5, row = 0, padx = 5, pady = 5, sticky = W)
divider(option_toolbar, 695).grid(column = 0, columnspan = 5, row = 1, padx = 5, pady = 5)

checkbuttons = [Checkbutton(option_toolbar, text = variable_title, variable = getattr(options, variable_name)) for variable_name, variable_title in variable_zip if type(eval(variable_name)) == bool]
for i, button in enumerate(checkbuttons):
    button.grid(column = i, row = 2, padx = 5, pady = 5)

# ─────────────────────────────────────────────────────────────────────┤
# Create int value option menus
# ─────────────────────────────────────────────────────────────────────┤

for column, title in enumerate(('Step size:      ','Window size:')):
    Label(option_toolbar, text = title).grid(column = 6+column, row = 0, padx = 5, pady = 5, sticky = W)
    divider(option_toolbar, 100).grid(column = 6+column, row = 1, padx = 5, pady = 5)

optionmenus = [OptionMenu(option_toolbar, getattr(options, variable_name), *option_range) for variable_name, option_range in (('step_size', range(2, 24, 2)), ('window_size', range(3, 25, 2)))]
for i, menu in enumerate(optionmenus):
    menu.grid(column = 6+i, row = 2, padx = 5, pady = 5, sticky = (E,W))

# ─────────────────────────────────────────────────────────────────────┤
# Create path value entry boxes
# ─────────────────────────────────────────────────────────────────────┤



for column, title in enumerate(('Input file location:', 'Output file location:', 'Scale file location:')):
    Label(option_toolbar, text = title).grid(column = 8+(column*2), columnspan = 2, row = 0, padx = 5, pady = 5, sticky = W)
    divider(option_toolbar, 215).grid(column = 8+(column*2), columnspan = 2, row = 1, padx = 5, pady = 5)

for i, variable_name in enumerate([variable_name for variable_name in variable_names if type(eval(variable_name)) == str and os.path.exists(eval(variable_name))]):
    Entry(option_toolbar, textvariable = getattr(options, variable_name)).grid(column = 8 + (i*2), row = 2, padx = 5, pady = 5)
    Button(option_toolbar, text = '⤶', command = lambda variable_name = variable_name: getattr(options, variable_name).set(filedialog.askdirectory())).grid(column = 9+(i*2), row = 2, padx = 5, pady = 5)

# ─────────────────────────────────────────────────────────────────────┤
# Set up option overrides and disables
# ─────────────────────────────────────────────────────────────────────┤

optionmenus[0].configure(state= DISABLED)
getattr(options, 'all_output_override').trace('w', lambda name, index, mode: [i.configure(state = {True:DISABLED, False:NORMAL}[options.all_output_override.get()])for i in checkbuttons[1:]])
getattr(options, 'all_output_override').trace('w', lambda name, index, mode: [i.select() if options.all_output_override.get() else i.deselect() for i in checkbuttons[1:]])
getattr(options, 'window_averaging').trace('w', lambda name, index, mode: optionmenus[0].configure(state = {True:NORMAL, False:DISABLED}[options.window_averaging.get()]))

# ─────────────────────────────────────────────────────────────────────┤
# Create button that initialises file output
# ─────────────────────────────────────────────────────────────────────┤

output_button = Button(option_toolbar, text = 'Output with these settings', command = produce_output, height = 4).grid(column = 14, row = 0, rowspan = 3, padx = 5, pady = 5)

# ─────────────────────────────────────────────────────────────────────┤
# Toolbar population END
# ─────────────────────────────────────────────────────────────────────┤

# ─────────────────────────────────────────────────────────────────────┤
# Create plotting buttons START
# ─────────────────────────────────────────────────────────────────────┤

refresh_directory = Button(file_browser, text = 'Search directory', width = 22, command = lambda: scan_output(0, output_location))
refresh_directory.grid(column = 0, row = 0, padx = 5, pady = 5)

purge_directory = Button(file_browser, text = 'Purge directory', width = 22, command = purge_directory)
purge_directory.grid(column=0, row=1, padx=5, pady=5)

plot_selected = Button(file_browser, text = 'Plot selected file', command = plot_selected, width = 22)
plot_selected.grid(column = 0, row = 2, padx = 5, pady = 5)

divider(file_browser, 200).grid(column = 0, row = 3, padx = 5, pady = 5)

# ─────────────────────────────────────────────────────────────────────┤
# Create plotting buttons END
# ─────────────────────────────────────────────────────────────────────┤

# ─────────────────────────────────────────────────────────────────────┤
# Check for matplotlib START
# ─────────────────────────────────────────────────────────────────────┤

try:
    import matplotlib.pyplot as plt
    plt.ion()
    font = {'family': 'Helvetica',
            'weight': 'regular',
            'size': 18}
    plt.rc('font', **font)
except ImportError:
    messagebox.showerror('Error!','Matplotlib not found, plotting functionality disabled')

# ─────────────────────────────────────────────────────────────────────┤
# Check for matplotlib END
# ─────────────────────────────────────────────────────────────────────┤

root.mainloop()
