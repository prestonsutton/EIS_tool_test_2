# -*- coding: utf-8 -*-
"""
test edit line (added June 06 2022) 

Created on Wed May 25 08:49:43 2022

authors: nicola.mazzanti and preston.sutton
TO DO
    -get back the time stamp and add it to the outputs, so that we can then do time evolution!
    -reinsert peak fit plots and Nyquist/Bode plots
    -check whether to run with outliers or without (see the warning it raises)
    -modify it so that it can be easily run via command line 
    -maybe add inserting a input/output  similar to what they did in Holyscript to select the data to run
    -IMPORTANT: we need to find a better datastructure to store the output


"""

#%% Functions
import glob
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date, time, datetime, timedelta
import csv

from impedance.models.circuits import  CustomCircuit
from impedance.visualization import plot_nyquist, plot_residuals
from impedance.validation import linKK
from impedance import preprocessing
from impedance.preprocessing import readGamry
from scipy.stats import chisquare 
from galvani import BioLogic 
from bayes_drt.inversion import Inverter
import bayes_drt.file_load as fl
import bayes_drt.plotting as bp

#%% data loading as nicola
def read_gamry_and_biologic (filename)    :
    from impedance.preprocessing import readGamry
    from galvani import BioLogic 
    from datetime import date, time, datetime, timedelta
    day_marker = 'DATE' #used to find the date of the measurement in .DTA files
    time_marker ='TIME' #used to find the time of the measurement in .DTA files

    if '.DTA' in filename:
        freq,Z = readGamry(filename) #load the file
        df_dta = pd.DataFrame()
        df_dta['freq/Hz'] = freq
        df_dta['Re(Z)/Ohm'] = Z.real
        df_dta['-Im(Z)/Ohm'] = -Z.imag
        df_dta['Phase(Z)/deg'] = np.degrees(np.arctan(Z.imag/Z.real))
        #now we get the timestamp and the start date        
        with open (filename, 'rt') as myfile:  # Open file for reading text
                for myline in myfile:
                   if myline.startswith(day_marker):
                       date_day = myline.split('\t')[2]
                   if myline.startswith(time_marker):
                       date_time = myline.split('\t')[2]
                       break
                myfile.close()
            
        date_complete =  date_day+ ' ' + date_time #time and date where the measure was taken
        format_string = "%d-%m-%Y %H:%M:%S" #format of the date
        start_time = datetime.strptime(date_complete, format_string)
        #test_time_for_biologic = R_ps = np.zeros(len(freq))
        #df_dta['time/s'] = test_time_for_biologic
        #timestamp = (start_time -datetime(1970, 1, 1))/timedelta(seconds = 1) #converting time in seconds elapsed since epoch

        return (freq, Z, df_dta, start_time)
        
    if '.mpr' in filename:
        mpr_file = BioLogic.MPRfile(filename)
        df_mpr = pd.DataFrame(mpr_file.data)
        freq = np.asarray(df_mpr['freq/Hz'], dtype = np.float64)
        Z_real = df_mpr['Re(Z)/Ohm']
        Z_imm_neg = df_mpr['-Im(Z)/Ohm']
        Z = np.asarray(df_mpr['Re(Z)/Ohm'] -1j*df_mpr['-Im(Z)/Ohm'], dtype = np.complex_)
        start_time= mpr_file.timestamp
        p = '%Y-%m-%d %H:%M:%S.%f'
        format_string = "%Y-%m-%d %H:%M:%S" #format of the date

        epoch = datetime(1970, 1, 1, 0, 0, 0)
        #start_time_epoch = (datetime.strptime(str(start_time), p) - epoch).total_seconds()
        start_time_epoch = (datetime.strptime(str(start_time), p) - epoch).total_seconds()
        corrected_start_time = start_time_epoch +df_mpr['time/s'][0]
        start_time = datetime.fromtimestamp(corrected_start_time).strftime(format_string)
        
        #df_dta['time/s'] = df_mpr['time/s']
        #timestamp = datetime.timestamp()
        return (freq, Z, df_mpr , start_time)
    
def collect_fitting_results( files, parameters, columns ='default', sample_name_position = 1, separator = '_'):
    '''
    loops through all the .txt files in the folder and to collect the fit results 
    in a pandas dataframe that is then returned.
    columns: the names of the columns of the datafolder
    Remember that you need to select the parameters of the model you have chosen.
    
    '''
    if columns == 'default':
        columns = parameters
        
    columns.append('sample_name') #add column to save samples name
    #create dataframe to store the data
    fitting_results = pd.DataFrame(columns = columns, index =files )
    
    #skip the initial lines
    start_line = 'Fit parameters:'
    
    #here we do a collection of the
    #listr = files
    for filename in files:
        sample_name = filename.split(separator)[sample_name_position]
        fitting_results['sample_name'][filename] = sample_name
        with open (filename, 'rt') as myfile:  # Open file for reading text
            for myline in myfile:
                if start_line in myline:
                    break
            for myline in myfile:            
                for param in parameters:
                    if param == 'start day and time' in myline:
                            print( myline)
                            par_value1 = myline.split('=')[1]
                            par_value = par_value1
                            print (par_value)
                            fitting_results[param][filename] = par_value
                    elif param in myline and 'string' not in myline:
                            par_value1 = myline.split('=')[1]
                            par_value = par_value1.split()[0]
                            fitting_results[param][filename] = par_value  
                    

                    
    return(fitting_results)



#%% definitions of all parameters, folder names and similar things

#lin kk parameters
output_linkk = 'output-linkk' #folder where it saves linkk test results. Use hyphens and NOT underscores!
c_param_real = 0.005 #threshold for lin kk
c_param_imm = 0.005 #threshold for lin kk
c_kk = 0.8 # over/under fitting evaluation in lin-kk, generally do not change it!

#getting the cell name from the files
separator = '_'
cell_position = 1

#parameters for the fitting  with Equivalent circuits
folder = 'fit_ECM_results_and_plots' #folder where the fit results will be saved
circ = 'L0-R0-p(R1,CPE1)-p(R2,CPE2)' #this is the equivalent circuit used for fitting
initial_guess=[1e-7, 0.01, 0.01, 0.0001, 0.7, 0.05, 0.0001, 0.7] #these are the initial guesses used for each element


#plotting parameters for equivalent circuit
leg_font_size = 'x-small' #font size of the legend in the plot, xx-small is the next smaller
loc_legend = 'best' #location of the legend within the plot
axes_label_size = 'xx-large' #font size of the axis
day_marker = 'DATE' #used to find the date of the measurement in .DTA files
time_marker ='TIME' #used to find the time of the measurement in .DTA files

#parameters for result collection, parameters should match your circ
folder_name = 'fit_results_ECM' #delete me
name_of_collected_res = 'fit_result_ECM_'+ folder_name  + '.tsv'
parameters = ['start day and time', 'L0','R0', 'R1', 'CPE1_0','CPE1_1', 'R2', 'CPE2_0', 'CPE2_1']


#DRT: location of files, and output folder of the DRT fits and peak fit results
#datadir = 'data_loop' #name input data file
data_output_DRT = 'output-DRT'  #folder, Use hyphens and NOT underscores!
DRT_plot_folder = 'DRT-plots'
DRT_peaks_folder = 'DRT-peaks-results'
peak_fit_filename = '_all-peak-fits' # Use hyphens and NOT underscores!
all_plots_figname = '_all-DRT' # Use hyphens and NOT underscores!
#creating the folders for DRT where we'll save the images

    
#%%DRT fit parameters

nonneg_optimized = False

prom_rthresh_optimized = 0.001

R_rthresh_optimized = 0.005

l2_penalty_optimized = 0.01

check_chi_sq_optimized = False

chi_sq_thresh_optimized = 0.4

chi_sq_delta_optimized = 0.2

outliers_mode = 'auto'

init_from_ridge_optimized = False

if init_from_ridge_optimized == True:
    ridge_kw_optimized = {'weights':'modulus'}
    
else:
    ridge_kw_optimized = {}
    
#%% collecting data and performing lin-kk test
directory_path = os.getcwd()
folder_name = os.path.basename(directory_path)

if not os.path.isdir(output_linkk):
    os.mkdir(output_linkk)



files_tested = [] #all files for which we did kk
good_kk_files = [] #files with satisfactory kk
failed_kk_files = [] #files with unsatisfactory kk
kk_test_results = pd.DataFrame(index = None) #dataframe where we'll save everything
chi2real_list = [] #list of real chi2
chi2imm_list= [] # list of imag chi2
good_chi2_values = [] #1/0 depending on whether the chi2 were satisfactory


allfiles = os.listdir() #all files in directory
#collect .DTA and .mpr files
for file in allfiles:
    if ('.DTA') in file or ('.mpr') in file:
        files_tested.append(file)

#read data from files, keep only data in 1st quadrant and do lin-kk test        
for filename in files_tested:           
    freq, Z, df_data, start_time = read_gamry_and_biologic(filename)    
    freq, Z = preprocessing.ignoreBelowX(freq, Z) #keep the data in the "first quadrant"
    conform_param_real = c_param_real
    conform_param_imm = c_param_imm
        
    #now we do the linkk test
    M, mu, Z_linKK, res_real, res_imag = linKK(freq, Z, c= c_kk, max_M= 100, fit_type = 'complex', add_cap=True)
    
    #calculate the chi real and imaginary, and printing it out
    try:
        chi2_real = sum((Z.real - Z_linKK.real)/abs(Z))/len(freq)
        print (filename +' chi2 real: ' + str (chi2_real))
        chi2real_list.append(chi2_real)
    except:
        print ('could not calculate chi on Zreal')
    
    try:
        chi2_imm = sum((Z.imag - Z_linKK.imag)/abs(Z))/len(freq)
        print (filename + ' chi2 imm: ' + str (chi2_imm) +'\n')
        chi2imm_list.append(chi2_imm)
    except:
        print ('could not calculate chi on Zimm')
    
    #verify that the real and imag chi2 satisfy the parameter, and then store their values       
    if chi2_real<conform_param_real and chi2_imm<conform_param_imm:
        good_kk_files.append(filename)
        good_chi2_values.append('Yes') # 1 will act a True switch to select the good quality data
    else:
        failed_kk_files.append(filename)
        good_chi2_values.append('No') # 0 will act a False switch to select the bad quality data

#save kk test results in a dataframe, that is then saved in .tsv    
kk_test_results['file'] = files_tested
kk_test_results['chi2 real'] = chi2real_list
kk_test_results['chi2 imm'] = chi2imm_list
kk_test_results['good chi'] = good_chi2_values
kk_test_results.set_index('file', inplace=True)
kk_test_results.to_csv(os.path.join(output_linkk, folder_name + '_kk-test-res.tsv'), sep = '\t')
    
#%% performing DRT fit on files that passed the kk-lin test
if not os.path.isdir(data_output_DRT):
    os.mkdir(data_output_DRT)
files = good_kk_files

#Create preset empty dataframe for data saving (these are were the system stores results, probably?)
#df = fl.read_eis(files[0]) commented out to use nicola's loader
#freq,Z = fl.get_fZ(df) #commented out to use nicola's loader
freq, Z, df, start_time = read_gamry_and_biologic (files[0])
#Z_raw = np.zeros((len(files),len(freq)),dtype='complex')
#Z_fit = np.zeros((len(files),len(freq)),dtype='complex')

peak_fits = [] #this will be a list with all the fit results

R_ps = np.zeros(len(files))
R_infs = np.zeros(len(files))
L = np.zeros(len(files))

chi_sq_DRT = np.zeros(len(files))
chi_sq_PeakFit = np.zeros(len(files))
start_times = []

#%DRT fitting code
#read what these lines do!
inv_check = Inverter() #what does this line do
inv_check.fit(freq, Z) # what is it?
#tauplot = inv_check.distributions['DRT']['tau']
#gamma = np.zeros((len(files),len(tauplot)))
#basis_freq = 1 / (2 * np.pi * tauplot)
peak_results = pd.DataFrame()
for i,file in enumerate(files):
    print('\n')
    print(file)
    fname = os.path.basename(file)
    name_root = os.path.splitext(file)[0] #get name of the file

    freq, Z, df, start_time= read_gamry_and_biologic (file)
    # Create Inverter instance ("optimize" = MAP; "sample" = HMC)
    # Set the basic frequency to match the measured frequency range
    # NOTE: Basic frequency must have 10 ppd
    inv = Inverter()
    inv.fit(freq, Z)
    tauplot = inv.distributions['DRT']['tau'] #added this to see if it solves the issues
    #gamma[i]=inv.predict_distribution("DRT") #I commented it out this when I modified the for loop
    gamma = inv.predict_distribution("DRT")
    R_ps[i] = inv.predict_Rp()
    R_infs[i] = inv.R_inf
    L[i] = inv.inductance
    chi_sq_DRT[i] = inv.score(freq, Z, weights='modulus')

    print('Chi-square DRT = {:.1e}'.format(chi_sq_DRT[i])) #store it somewhere?
    
    # Once we have the DRT fit, we perform a peak fit on it 
    inv.fit_peaks(prom_rthresh = prom_rthresh_optimized,
    R_rthresh = R_rthresh_optimized,
    l2_penalty = l2_penalty_optimized,
    check_chi_sq = check_chi_sq_optimized,
    chi_sq_thresh = chi_sq_thresh_optimized,
    chi_sq_delta = chi_sq_delta_optimized)
    peak_fit = inv.extract_peak_info()
    
    print('Chi-square PeakFit = {:.1e}'.format(peak_fit['chi_sq'])) 
    chi_sq_PeakFit[i] = peak_fit['chi_sq']
    peak_fit['freq'] = 1 / (2 * np.pi * peak_fit['tau_0'])
    peak_fit['file'] = name_root #used for saving
    peak_fit['start_time'] = start_time #used for saving

    peak_fit_results =pd.DataFrame.from_dict(peak_fit).transpose() #outputfile for fit
    peak_fit_results.drop('num_peaks', inplace=True) #removing the info on how many peaks there are
    peak_fits.append(peak_fit)
    start_times.append(start_time)

    
    inv.save_fit_data(os.path.join(data_output_DRT, name_root +'.pkl'),which='core') # saving fit results to a pickle
    peak_fit_results.to_csv(os.path.join(data_output_DRT,name_root +"_DRT-peaks-results.tsv"),index=True, sep = '\t')     #save peaks to a tsv

    #exporting drt data
    csv_array = np.vstack((tauplot,gamma))
    columns = ['tau'] + [os.path.basename(file) ]
    drt_out = pd.DataFrame(csv_array.T,columns = columns)
    drt_out.to_csv(os.path.join(data_output_DRT,name_root +"_DRT-DATA.tsv"),index=True, sep = '\t') #save taus and Rp in a tsv   
#%% saving pathfile
with open(os.path.join(data_output_DRT,folder_name + peak_fit_filename +"peak-fits-all.tsv"), 'w', encoding='utf8', newline='') as output_file:
#with open('ciao' +"_peak_fits_all.tsv", 'w', encoding='utf8', newline='') as output_file:

    fc = csv.DictWriter(output_file, 
                       fieldnames=peak_fits[0].keys(), delimiter = '\t')
    fc.writeheader()
    fc.writerows(peak_fits)

#%% fitting the DRT plots
fit_files = glob.glob(os.path.join(data_output_DRT,'*.pkl'))
#fit_files = sorted(fit_files,key=os.path.getmtime) commented it out as the modification time is not meaningful


#lists to store info on how to plot   
cell_list_all = []
#cycle_list_all = []
maps_for_plots = []
markers_shared = []
plot_info = pd.DataFrame()
plot_info['files'] = fit_files

#getting the cell and cycles from the filenames   
for filename in fit_files:  
        cell_name = filename.split(separator)[cell_position]
        cell_list_all.append(cell_name)
        files.append(filename)

plot_info['cell_label'] = cell_list_all
plot_info['start_time'] = start_times


#all the files on the same plot
fig,ax = plt.subplots()#create the figure
inv_DRT = Inverter()

for file, start_time in zip(fit_files, start_times):
    #name_root = os.path.splitext(file)[0] #get name of the file

    inv_DRT.load_fit_data(file)
    label =  file.split(separator)[cell_position] + ' ' + str(start_time)

    #label = '{}_{}_{}'.format(file.split('_')[4],file.split('_')[5],file.split('_')[7])
    inv_DRT.plot_distribution(ax=ax,normalize=False, label = label)

ax.legend(fontsize='x-small', bbox_to_anchor=(1, 1), loc='upper left', facecolor = 'w')
fig.tight_layout()
fig.savefig(os.path.join(data_output_DRT,  folder_name + all_plots_figname + ".png"), dpi = 250)      

#list of the unique cells and cycles present
cell_list = list(set(cell_list_all))
#cycle_list = list(set(cycle_list_all))

for cell in cell_list:
    fig1, ax1 = plt.subplots()#create the figure
    plot_info_cell =    plot_info[plot_info['cell_label']==cell]
    for  file, start_time  in  zip(plot_info_cell['files'], plot_info_cell['start_time'] ):  

        inv_DRT.load_fit_data(file)
        label =  file.split(separator)[cell_position] + ' ' + str(start_time)

        #label = '{}_{}_{}'.format(file.split('_')[4],file.split('_')[5],file.split('_')[7])
        inv_DRT.plot_distribution(ax=ax1,normalize=False, label = label)
        
    ax1.legend(fontsize='x-small', bbox_to_anchor=(1, 1), loc='upper left', facecolor = 'w')
    fig1.tight_layout()
    fig1.savefig(os.path.join(data_output_DRT, folder_name + '_' + str(cell) + ".png"), dpi = 250)
    
plt.close('all')


#%% fitting the files with equivalent circuits

if not os.path.isdir(folder):
    os.mkdir(folder)

files_good_kk = kk_test_results.loc[kk_test_results['good chi'] == "Yes"]
#here we do the fitting
for i,file in enumerate(files_good_kk.index):
    print('\n')
    print(file)
    fname = os.path.basename(file)
    
    name_root = os.path.splitext(file)[0] #get name of the file

    freq, Z, df, start_time= read_gamry_and_biologic (file)
    #performing the fit and calculating the residuals
    circuit = CustomCircuit(initial_guess=initial_guess,  circuit=circ)
    circuit.fit (freq, Z)
    customCircuit_fit = circuit.predict(freq)
    Z_fit = circuit.predict(freq) #simulated values
    res_meas_real = (Z - circuit.predict(freq)).real/np.abs(Z) #residuals
    res_meas_imag = (Z - circuit.predict(freq)).imag/np.abs(Z) #residuals
    
    #plotting the Nyquist of data and fit, and the residuals vs freq
    fig, ax = plt.subplots(nrows=2) #one is the Nyquist of the fit, the other one is the residuals vs the freq
    plot_nyquist(ax[0], customCircuit_fit, fmt='-', label = 'fit') # Nyquist fit
    plot_nyquist(ax[0], Z, label = 'data') #Nyquist data
    ax[0].axhline(y=0, color='k', linestyle='--') #adding the y=0 lines
    
    plot_residuals(ax[1], freq, res_meas_real, res_meas_imag) #plotting residuals

    #adjusting the plots
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    ax[0].grid(False)
    ax[0].set_xlabel('Z$_{Re}$ [$\Omega$]', fontsize = axes_label_size)
    ax[0].set_ylabel('-Z$_{Im}$ [$\Omega$]', fontsize = axes_label_size)
    ax[0].axis('equal') 
    ax[0].legend(loc = loc_legend, fontsize = leg_font_size) 
    ax[1].legend( fontsize = leg_font_size) 

    #note: understand get_figure, and why it gives an error when not using an index
    fig = ax[0].get_figure()
    fig.set_tight_layout(True) 
    #saving the plots in the specified folders
    fig.savefig(folder + '\\' + name_root + '_fit-nyquist.png')
    plt.show()
    #saving the fit results in a text file that is also placed in the specified folder
    print(circuit, '\n --------- \n',  '\n start day and time= ' + str(start_time),  file=open( folder + '\\' +name_root + ".txt", "w"))
    print (circuit)
    

plt.close ('all') #closing all the plots 

#%% collecting the fitting results
os.chdir(folder)

listr= []
for filename in glob.glob('*.txt'):
    listr.append(filename)

fitting_results_numeric = collect_fitting_results(parameters = parameters, files = listr)

fitting_results_numeric.to_csv(name_of_collected_res, sep = '\t')

os.chdir('..')
