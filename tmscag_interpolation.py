# DESCRIPTION:
# Input datasets:
#  (1) ENVI style image stack containing available fractional snow covered area derived from TMSCAG algorithm (currently 410 scenes)
#  (2) ENVI style image stack containing no data (clouds, gap-lines) for the same spatial domain and dates as the fractional snow covered area stack
#  (3) .csv file containing Landsat scene identifiers, which can be used to extract the scene date and other information

# output results for each date in the set of validdation bands, based on linear interpolation between valid, cloud-free fSCA
# interpolates between future and past dates, so would not be usable for real time or near real time applications

ns_min = 450
ns_max = 600

nl_min = 450
nl_max = 600

# import external libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import os
from osgeo import gdal
from osgeo.gdalconst import *
import numpy
from numpy import arange
from matplotlib import pyplot
import pandas
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix


# indicate whether or not to replace the mask values in all validation bands with 255 (no data value)
# do this when using the validation bands for accuracy assessment

# do not do this when attempting to output the best possible result
mask_validation_bands = 0

# decide whether or not to interpolate for the entire 11 year time series
# if this flag is set to zero, a subset time series (currently set to 2014-2015) will be interpolated
process_full_time_series = 1
process_subset_time_series = 0

solar_array_doy_values = [274,284,294,304,314,324,334,344,354,364,9,19,29,39,49,59,69,79,89,99,109,119,129,139,149,159,169,179,189]

# set number of validation bands
n_val_bands = 100

# indicate whether or not to output the full time series dataset (5000+ bands)
output_full_datasets = 1 

# set working directory and change to it
working_directory = '/Volumes/Snow_02/unbelievably_sheepish/ready/'
os.chdir(working_directory)

# load day of year information from lookup table saved as .csv file
doy_data = np.genfromtxt('np_doy_2000_2015.csv', delimiter=',', dtype=None)

#doy_data = pandas.read_csv('doy_2000_2015.csv', sep=',',header=1)

print doy_data
full_doy_array = doy_data
print(full_doy_array)

# import list of landat scenes ids to be used for interpolation
# currently working with scenes from 2004-2015
bands_dataset = np.genfromtxt('landsat_scenes.csv',delimiter=',', dtype=None)
bands_array = bands_dataset[:]
n_bands = len(bands_array)
day_of_year_array = np.zeros(n_bands)
year_array = np.zeros(n_bands)
days_from_start_array = np.zeros(n_bands)

# populate arrays indicating year, day of year, and days since January 1, 2000
start_day_of_year_array = [0,366,731,1096,1461,1827,2192,2557,2922,3288,3653,4018,4383,4749,5114,5479]
start_year_array = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]
for u in range(n_bands):
   # extract day of year and year from list of landsat scenes
   temp = bands_array[u]
   day_of_year = int(temp[21:24])
   year = int(temp[8:12])
   day_of_year_array[u] = day_of_year
   year_array[u] = year
   days_from_start_array[u] = start_day_of_year_array[year-2000] + day_of_year



# define file names that contain the stack of tmscag fSCA data and the stack of cloud/no data mask data
filenames_array = ['ff_0a_tmscag_stack_filled', 'ff_0a_mask_stack_filled','ff_0b_solar_stack']
image_names_array = ['tmscag_fsca_image','mask_image','solar_image']

# import images using GDAL
filename = filenames_array[0]
tmscag_fsca_image = gdal.Open(filename, gdal.GA_ReadOnly)
rows = tmscag_fsca_image.RasterYSize
cols = tmscag_fsca_image.RasterXSize
tmscag_stack_array = tmscag_fsca_image.ReadAsArray()
print('Opened ', filename)

filename = filenames_array[1]
mask_image = gdal.Open(filename, gdal.GA_ReadOnly)
mask_stack_array = mask_image.ReadAsArray()
print('Opened ', filename)

filename = filenames_array[2]
solar_image = gdal.Open(filename, gdal.GA_ReadOnly)
solar_stack_array = solar_image.ReadAsArray()
print('Opened ', filename)

# extract the dimensions from the stack (need to be identical for tmscag fsca and mask stacks)
dimensions = tmscag_stack_array.shape
#oldest_validation_bands_array = [6,39,49,55,65,102,118,130,137,160,166,179,185,193,205,213,232,258,264,275,301,309,351,375,394]
#old_validation_bands_array = [38,48,55,75,199,200,210,211,217,218,234,235,251,252,253,258,259,260,277,278,279,284,285,286]

# identify bands for exclusion that will later be used for validation
# 100 bands have been randommly selected for exclusion
# these numbers correaspond to the band number in the tmscag stack (and cloud mask arrays)
# for example, 0 would be the first band in the stack, which happens to be the scene aquired 3/8/2004
validation_bands_array = [346,133,45,113,258,302,18,187,67,132,168,31,22,216,87,223,368,337,205,81,143,321,124,312,200,265,338,152,177,104,370,385,376,262,363,
   34,350,300,408,12,254,391,166,51,146,294,134,267,169,375,39,393,360,116,26,235,279,345,7,243,343,225,290,16,112,156,342,1,154,9,379,10,
   214,43,196,256,311,276,320,4,191,2,314,114,128,105,139,173,210,75,147,42,195,29,83,36,355,131,382,88]

# identify the day of year for each validation band
# this can be used to subset validation results by season (e.g. only ouput day of year values 1-31, for January results)
for tb in range(n_val_bands):
   band_value = validation_bands_array[tb]
   print(day_of_year_array[band_value])


# build an array that contains the position for each validation band in the much larger array
# that spans all days between 1/1/2000 and 12/31/2015
# this will later be used to output interpolation results from these days
val_bands_position_array = days_from_start_array[validation_bands_array]
validation_bands_position_array = numpy.int16(val_bands_position_array)


# transfer data from the excluded bands to a separate validation stack and output as a separate file
# do this for both the mask stack and the tmscag fsca stack
mask_validation_stack = np.zeros((n_val_bands,dimensions[1],dimensions[2]))
mask_validation_stack = mask_stack_array[validation_bands_array,:,:]

tmscag_validation_stack = np.zeros((n_val_bands,dimensions[1],dimensions[2]))
tmscag_validation_stack = tmscag_stack_array[validation_bands_array,:,:]

# NEW OUTPUT FILE
driver = tmscag_fsca_image.GetDriver()
validation_mask_stack_outname = (working_directory + '/v6_tmscag_validation_mask_stack')
outDs = driver.Create(validation_mask_stack_outname, cols, rows, n_val_bands, GDT_Float32)

if outDs is None:
    print 'Could not create file'
    sys.exit(1)

for out_band_number in range(n_val_bands):
   outBand = outDs.GetRasterBand(out_band_number+1)
   outData = mask_validation_stack[out_band_number, : , : ]
   outBand.WriteArray(outData, 0, 0)
   # flush data to disk, set the NoData value and calculate stats
   outBand.FlushCache()
   outBand.SetNoDataValue(-99)
# END OF NEW OUTPUT FILE

# NEW OUTPUT FILE   
file_out_path_x = (working_directory + '/v6_tmscag_validation_stack')
outDs = driver.Create(file_out_path_x, cols, rows, n_val_bands, GDT_Float32)

if outDs is None:
    print 'Could not create file'
    sys.exit(1)

for out_band_number in range(n_val_bands):
   outBand = outDs.GetRasterBand(out_band_number+1)
   outData = tmscag_validation_stack[out_band_number, : , : ]
   outBand.WriteArray(outData, 0, 0)
    
   # flush data to disk, set the NoData value and calculate stats
   outBand.FlushCache()
   outBand.SetNoDataValue(-99)
# END OF NEW OUTPUT FILE

# build the interpolated tmscag cube, which contains daily fSCA values
# for all days between January 1, 2000 and December 31, 2015
# days between January 1, 2000 and December 31, 2003 will not be filled

interpolated_tmscag_cube = np.zeros((1553,dimensions[1],dimensions[2])) # create a cube that holds all days between Jan 1 2000 and Dec 31 2015
subset_interpolated_tmscag_cube = np.zeros((730,dimensions[1],dimensions[2])) 

# build an offset cube which indicates, for each pixel, how many days away (the offset)
# it is from a valid cloud-free observation which resulted in valid TMSCAG fsca
interpolated_tmscag_offset_cube = np.zeros((1553,dimensions[1],dimensions[2]))
subset_interpolated_tmscag_offset_cube = np.zeros((730,dimensions[1],dimensions[2]))



# now that the original data values and mask values have been transferred to
# a separate validation mask stack, set all values in the original mask stack
# to 255 for all of the excluded validation bands
for valband in range(n_val_bands):
   validation_band = validation_bands_array[valband]
   print('Band ', validation_band, ' excluded')
   if mask_validation_bands == 1:
      mask_stack_array[validation_band,:,:] = 255


# for each pixel, interpolate between valid observations to create a continuous daily snow cover time series
series_result = [0]
#for i in range(dimensions[1]):
for i in range(ns_min,ns_max):
   i_mod = i % 10
   print(i)
   #for j in range(dimensions[2]):
   for j in range(ns_min,ns_max):
      j_mod = j % 100
      mask = mask_stack_array[:,i,j]
      tmscag_fsca = tmscag_stack_array[:,i,j]
      valid_obs_array = np.zeros(5844)

      vc = 0
      n = len(mask)
      for g in range(n_bands):
         if mask[g] != 2 and mask[g] != 255:
            vc = vc+1
      
      # initialize fsca time series with empty values           
      fsca_series = np.empty(vc)
      day_series = np.empty(vc)
      
      a = 0
      for g in range(n_bands):
         if mask[g] != 2 and mask[g] != 255:
            fsca_series[a] = tmscag_fsca[g]
            day_series[a] = days_from_start_array[g]
            # add a presence value
            valid_obs_array[days_from_start_array[g]] = 1
            a = a+1
 
      # identify the days for which to implement interpolation
      # could cut this to a subset of the full number of days to speed up the process
      if process_full_time_series == 1:
         
         #days_to_interpolate = range(5844)
         #offset_array = range(5844)
         
         sd = 2830
         days_to_interpolate = range(2830,4383)
         offset_array = range(2830,4383)
         doy_doy_array = np.empty((4383-2830))
         a = 0
         for dd in range(2830,4383):
            doy_doy_array[a] = full_doy_array[dd]
            a = a+1
             
         xp = day_series
         fp = fsca_series
         # temporal interpolation for the entire time series
         series_result = np.interp(days_to_interpolate,xp,fp)
      
         # for each day in the interpolated time series, calculate the minimum distance
         # (in number of days) to a valid, cloud-free TMSCAG value
         for r in range(len(days_to_interpolate)):
            arr = np.array(day_series)
            arr_diff = r - arr
            arr_diff_abs = abs(arr_diff)
            offset_array[r] = min(arr_diff_abs)

         # go through and set values < 200 to zero
         # initial analysis indicates most interpolated values < 200 corresponded to true values of 0
         series_length = len(series_result)
         for r in range(series_length):
            if series_result[r] <= 200:
               series_result[r] = 0  
      
         interpolated_tmscag_cube[:,i,j] = series_result
         interpolated_tmscag_offset_cube[:,i,j] = offset_array
      
      #year_2014_2015_result 
      if process_subset_time_series == 1:
         subset_days_to_interpolate = range(730)
         for r in range(730):
            # add 5114 to start with January 1, 2014
            subset_days_to_interpolate[r] = r + 5114
         subset_offset_array = range(730)
      
         xp = day_series
         fp = fsca_series
         # temporal interpolation for the entire time series
         subset_series_result = np.interp(subset_days_to_interpolate,xp,fp)
      
         for r in range(len(subset_days_to_interpolate)):
            arr = np.array(day_series)
            r_0 = r + 5114
            arr_diff = r_0 - arr
            arr_diff_abs = abs(arr_diff)
            subset_offset_array[r] = min(arr_diff_abs)

         # go through and set values < 200 to zero
         # initial analysis indicates most interpolated values < 200 corresponded to true values of 0
         subset_series_length = len(subset_series_result)
         for r in range(subset_series_length):
            if subset_series_result[r] <= 200:
               subset_series_result[r] = 0
         
         #if (i_mod == 0) and (j_mod == 0):
         #   print(subset_series_result)
         subset_interpolated_tmscag_cube[:,i,j] = subset_series_result
         subset_interpolated_tmscag_offset_cube[:,i,j] = subset_offset_array
      # feed the interpolated time series and the offset days time series out to the appropriate
      # location in the full sized cube


# now that the downscaling has been completed
# extract the interpolated results corresponding to the original validation scenes


tmscag_validation_predictions_stack = np.zeros((n_val_bands,dimensions[1],dimensions[2]))
tmscag_validation_predictions_offset_stack = np.zeros((n_val_bands,dimensions[1],dimensions[2]))

#validation_predictions_bands_array = numpy.array(validation_predictions_bands_array) + 1
#tmscag_validation_predictions_stack = interpolated_tmscag_cube[validation_bands_position_array,:,:]
#tmscag_validation_predictions_offset_stack = interpolated_tmscag_offset_cube[validation_bands_position_array,:,:]

# only output the full datasets (5000+ bands if flag is set to 1 at the beginning)
if output_full_datasets == 1:
   
   # NEW OUTPUT FILE
   print 'Writing out full interpolated time series cube...'
   driver = tmscag_fsca_image.GetDriver()
   file_out_path = (working_directory + '/v6a_interpolation_output')
   outDs = driver.Create(file_out_path, cols, rows, 1553, GDT_Float32)

   if outDs is None:
      print 'Could not create file'
      sys.exit(1)

   for out_band_number in range(0,1553):
     print('out band', out_band_number)

     outBand = outDs.GetRasterBand(out_band_number+1)
     outData = interpolated_tmscag_cube[out_band_number, : , : ]
     outBand.WriteArray(outData, 0, 0)

     # flush data to disk, set the NoData value and calculate stats
     outBand.FlushCache()
     outBand.SetNoDataValue(-99)
     # END OF NEW OUTPUT FILE
   

   # NEW OUTPUT FILE
   print 'Writing out full interpolated time series cube offset values...'
   file_out_path_b = (working_directory + '/v6a_interpolation_output_offset')
   outDs = driver.Create(file_out_path_b, cols, rows, 1553, GDT_Int32)

   if outDs is None:
      print 'Could not create file'
      sys.exit(1)

   for out_band_number in range(0,1553):
     #print('out band', out_band_number)

     outBand = outDs.GetRasterBand(out_band_number+1)
     outData = interpolated_tmscag_offset_cube[out_band_number, : , : ]
     outBand.WriteArray(outData, 0, 0)
    
     # flush data to disk, set the NoData value and calculate stats
     outBand.FlushCache()
     outBand.SetNoDataValue(-99)   
   # END OF NEW OUTPUT FILE

# OUTPUT 2014-2015 time series file
driver = tmscag_fsca_image.GetDriver()
file_out_path = (working_directory + '/final_v6_2014_2015_interpolation_output')
outDs = driver.Create(file_out_path, cols, rows, 730, GDT_Float32)

if outDs is None:
   print 'Could not create file'
   sys.exit(1)

for out_band_number in range(0,730):
   print('out band', out_band_number)
   outBand = outDs.GetRasterBand(out_band_number+1)
   outData = subset_interpolated_tmscag_cube[out_band_number, : , : ]
   outBand.WriteArray(outData, 0, 0)
   # flush data to disk, set the NoData value and calculate stats
   outBand.FlushCache()
   outBand.SetNoDataValue(-99)
# END OUTPUT FILE

# OUTPUT OFFSET STACK
driver = tmscag_fsca_image.GetDriver()
file_out_path = (working_directory + '/final_v6_2014_2015_interpolation_offset_output')
outDs = driver.Create(file_out_path, cols, rows, 730, GDT_Float32)

if outDs is None:
   print 'Could not create file'
   sys.exit(1)

for out_band_number in range(0,730):
   print('out band', out_band_number)

   outBand = outDs.GetRasterBand(out_band_number+1)
   outData = subset_interpolated_tmscag_offset_cube[out_band_number, : , : ]
   outBand.WriteArray(outData, 0, 0)
   # flush data to disk, set the NoData value and calculate stats
   outBand.FlushCache()
   outBand.SetNoDataValue(-99)
# END OF OUTPUT FILE


# NEW OUTPUT FILE
print 'Writing out interpolated results for validation stack...'
file_out_path_3 = (working_directory + '/v6_tmscag_validation_interp_prediction_stack')
outDs = driver.Create(file_out_path_3, cols, rows, n_val_bands, GDT_Float32)

if outDs is None:
    print 'Could not create file'
    sys.exit(1)

for out_band_number in range(n_val_bands):
   outBand = outDs.GetRasterBand(out_band_number+1)
   outData = tmscag_validation_predictions_stack[out_band_number, : , : ]
   outBand.WriteArray(outData, 0, 0)
    
   # flush data to disk, set the NoData value and calculate stats
   outBand.FlushCache()
   outBand.SetNoDataValue(-99)
# END OF NEW OUTPUT FILE

# NEW OUTPUT FILE
print 'Writing out offset values for interpolated validation stack...'
file_out_path_4 = (working_directory + '/v6_tmscag_validation_prediction_offset_stack')
outDs = driver.Create(file_out_path_4, cols, rows, n_val_bands, GDT_Int32)

if outDs is None:
    print 'Could not create file'
    sys.exit(1)

for out_band_number in range(n_val_bands):
   outBand = outDs.GetRasterBand(out_band_number+1)
   outData = tmscag_validation_predictions_offset_stack[out_band_number, : , : ]
   outBand.WriteArray(outData, 0, 0)
    
   # flush data to disk, set the NoData value and calculate stats
   outBand.FlushCache()
   outBand.SetNoDataValue(-99)
# END OF NEW OUTPUT FILE


# post processing step 1
# smooth the time series using a 9x9 temporal window
smoothed_tmscag_cube = interpolated_tmscag_cube
for i in range(ns_min,ns_max):
   i_mod = i % 10
   print(i)
   #for j in range(dimensions[2]):
   for j in range(ns_min,ns_max):
      series = interpolated_tmscag_cube[:,i,j]
      n_days = series.size
      smoothed_series = series
      for d in range(4,(n_days-4)):
         t_array = [series[d-4],series[d-3],series[d-2],series[d-1],series[d],series[d+1],series[d+2],series[d+3],series[d+4]]
         smoothed_series[d] = np.mean(t_array)
      smoothed_tmscag_cube[:,i,j] = smoothed_series

# NEW OUTPUT FILE
print 'Writing out smoothed interpolate tmscag stack..'
file_out_path_4 = (working_directory + '/v6_smoothed_tmscag_stack')
outDs = driver.Create(file_out_path_4, cols, rows, n_days, GDT_Int32)

if outDs is None:
    print 'Could not create file'
    sys.exit(1)

for out_band_number in range(n_days):
   outBand = outDs.GetRasterBand(out_band_number+1)
   outData = smoothed_tmscag_cube[out_band_number, : , : ]
   outBand.WriteArray(outData, 0, 0)
    
   # flush data to disk, set the NoData value and calculate stats
   outBand.FlushCache()
   outBand.SetNoDataValue(-99)
# END NEW OUTPUT FILE

# extract every 5th day in time series
n_5_day_intervals = (n_days/5)+1
interval_5_tmscag_cube = np.zeros((n_5_day_intervals,dimensions[1],dimensions[2]))
smoothed_tmscag_cube = interpolated_tmscag_cube
for i in range(nl_min,nl_max):
   i_mod = i % 10
   print(i)
   #for j in range(dimensions[2]):
   for j in range(ns_min,ns_max):
      series = smoothed_tmscag_cube[:,i,j]
      interval_series = np.zeros(n_5_day_intervals)
      doy_series = np.zeros(n_5_day_intervals)
      interval = 0
      for d in range(n_days):
         d_mod = d % 5
         if d_mod == 0:
            interval_series[interval] = series[d]
            doy_series[interval] = doy_doy_array[d]
            interval = interval+1
      interval_5_tmscag_cube[:,i,j] = interval_series

print 'DOY SERIES'      
print(doy_series)

# NEW OUTPUT FILE
print 'Writing out smoothed interpolate tmscag stack..'
file_out_path_4 = (working_directory + '/v66_5_day_interval_output_stack')
outDs = driver.Create(file_out_path_4, cols, rows, n_5_day_intervals, GDT_Int32)

if outDs is None:
    print 'Could not create file'
    sys.exit(1)

for out_band_number in range(n_5_day_intervals):
   outBand = outDs.GetRasterBand(out_band_number+1)
   outData = interval_5_tmscag_cube[out_band_number, : , : ]
   outBand.WriteArray(outData, 0, 0)
    
   # flush data to disk, set the NoData value and calculate stats
   outBand.FlushCache()
   outBand.SetNoDataValue(-99)
# END NEW OUTPUT FILE

solar_interval_cube = np.zeros((1553,dimensions[1],dimensions[2]))
illumination_corrected_interval_5_tmscag_cube = interval_5_tmscag_cube

# apply post processing to fill in days when/where snow cover was missed because of poor illumination
for i in range(nl_min,nl_max):
   i_mod = i%100
   for j in range(ns_min,ns_max):
      j_mod = j%100
      for b in range(n_5_day_intervals):
         flag = 0
         #if interval_5_tmscag_cube[b,i,j] < 1000:
            # based on day of year, lookup solar radiation for this pixel
            # find the band position in the solar stack array that is closest to the day of year
         for sday in range(len(solar_array_doy_values)):
            difference = doy_series[b] - solar_array_doy_values[sday]
            if abs(difference) <= 6:
               flag = 1
               break
            #sday is now the array id for
            
         if flag == 1:
            solar_rad = solar_stack_array[sday,i,j]
            
            
         if flag == 0:
            solar_rad = 2999
         #if i_mod == 0 and j_mod == 0:
         #   print('------')
         #   print(i,j,b,solar_rad)
         #   print('------')
         solar_interval_cube[b,i,j] = solar_rad
         
         #if solar_rad < 2000:
            # if solar radiation < 2000, then look at nearby pixels

test_image = solar_interval_cube
test_image_2 = interval_5_tmscag_cube
solar_method = interval_5_tmscag_cube*0
for i in range(nl_min,nl_max):
   print(i)
   for j in range(nl_min,nl_max):
      for b in range(n_5_day_intervals):
         if solar_interval_cube[b,i,j] >= 2000:
            illumination_corrected_interval_5_tmscag_cube[b,i,j] = interval_5_tmscag_cube[b,i,j]
            solar_method[b,i,j] = 1
         # if solar radiation for the period < 2000 AND the day of year is between 315-366, or 1-120
         # then look to fill in values based on nearby pixels
         # doing this in the early season (prior to day 315) is likely to create false positives for snow cover
         # doing this after day 120 should not be necessary due to higher solar elevation angles (resulting in better illumination)
         if solar_interval_cube[b,i,j] < 2000 and interval_5_tmscag_cube[b,i,j] < 600: #and (doy_series[b] > 315) or (doy_series[b] < 120)):
            # next check to see if there is snow cover later in the season at this pixel
            # first determine the maximum number of days to look ahead and see if there is snow cover at that pixel, based on the day of year (doy)
            # divide by 5 because we are currently working with a 5 day interval data set
            if doy_series[b] < 120:
               max_forward_intervals_to_consider = (120-doy_series[b])/5
            if doy_series[b] >= 315:
               max_forward_intervals_to_consider = (365-doy_series[b]+119)/5
            
            max_forward_intervals_to_consider = int(max_forward_intervals_to_consider)
            snow_later_same_year = 0
            #for bb in range(b,b+max_forward_intervals_to_consider):
            #   if interval_5_tmscag_cube[bb,i,j] > 500:
            #      snow_later_same_year = 1
            #      break
            
            # only continue the process of filling in snow in poorly illuminated areas if snow is found at the same pixel later the same year
            #if snow_later_same_year == 1:
            
            # look at nearby pixels
            local_solar_array = []
            local_tmscag_array = []
            count = 0
            for ii in range(i-5,i+5):
               for jj in range(j-5,j+5):
                  if solar_interval_cube[b,ii,jj] > 2700:
                     #local_solar_array = np.append(local_solar_array,solar_interval_cube[b,ii,jj])
                     local_tmscag_array = np.append(local_tmscag_array,interval_5_tmscag_cube[b,ii,jj])
                     count = count+1
            
            if count > 11:
               tmscag_median = np.median(local_tmscag_array)
               test_image[b,i,j] = count  
               illumination_corrected_interval_5_tmscag_cube[b,i,j] = tmscag_median
               solar_method[b,i,j] = 2
            if count <= 11:
               local_tmscag_array = []
               count = 0
               for ii in range(i-7,i+7):
                  for jj in range(j-7,j+7):
                     if solar_interval_cube[b,ii,jj] > 2700:
                        local_tmscag_array = np.append(local_tmscag_array,interval_5_tmscag_cube[b,ii,jj])
                        count = count+1
               if count > 22:
                  tmscag_median = np.median(local_tmscag_array)
                  illumination_corrected_interval_5_tmscag_cube[b,i,j] = tmscag_median
                  solar_method[b,i,j] = 3
               if count <=22:
                  local_tmscag_array = []
                  count = 0
                  for ii in range(i-9,j+9):
                     for jj in range(j-9,j+9):
                        if solar_interval_cube[b,ii,jj] > 2700:
                           local_tmscag_array = np.append(local_tmscag_array,interval_5_tmscag_cube[b,ii,jj])
                           count = count+1
                  if count > 36:      
                      tmscag_median = np.median(local_tmscag_array)
                      illumination_corrected_interval_5_tmscag_cube[b,i,j] = tmscag_median
                      solar_method[b,i,j] = 4
                  if count <= 36:
                      local_tmscag_array = []
                      count = 0
                      for ii in range(i-17,j+17):
                          for jj in range(j-17,j+17):
                               if solar_interval_cube[b,ii,jj] > 2700:
                                  local_tmscag_array = np.append(local_tmscag_array,interval_5_tmscag_cube[b,ii,jj])
                                  count = count+1
                      if count > 100:
                         tmscag_median = np.median(local_tmscag_array)
                         illumination_corrected_interval_5_tmscag_cube[b,i,j] = tmscag_median
                         solar_method[b,i,j] = 4
                      if count <= 100:
                         interval_5_tmscag_cube[b,i,j] = interval_5_tmscag_cube[b,i,j]
                         solar_method[b,i,j] = 1
   #doy_series
   #solar_array_doy_values
   #solar stack array
   
# NEW OUTPUT FILE
print 'Writing out smoothed interpolate tmscag stack..'
file_out_path_4 = (working_directory + '/v66_solar_method_cube')
outDs = driver.Create(file_out_path_4, cols, rows, n_5_day_intervals, GDT_Int32)

if outDs is None:
    print 'Could not create file'
    sys.exit(1)

for out_band_number in range(n_5_day_intervals):
   outBand = outDs.GetRasterBand(out_band_number+1)
   outData = solar_method[out_band_number, : , : ]
   outBand.WriteArray(outData, 0, 0)
    
   # flush data to disk, set the NoData value and calculate stats
   outBand.FlushCache()
   outBand.SetNoDataValue(-99)
# END NEW OUTPUT FILE


# NEW OUTPUT FILE
print 'Writing out smoothed interpolate tmscag stack..'
file_out_path_4 = (working_directory + '/v66_illumination_corrected_interval_results')
outDs = driver.Create(file_out_path_4, cols, rows, n_5_day_intervals, GDT_Int32)

if outDs is None:
    print 'Could not create file'
    sys.exit(1)

for out_band_number in range(n_5_day_intervals):
   outBand = outDs.GetRasterBand(out_band_number+1)
   outData = illumination_corrected_interval_5_tmscag_cube[out_band_number, : , : ]
   outBand.WriteArray(outData, 0, 0)
    
   # flush data to disk, set the NoData value and calculate stats
   outBand.FlushCache()
   outBand.SetNoDataValue(-99)
# END NEW OUTPUT FILE