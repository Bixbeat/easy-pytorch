from . import image_manipulations as i_manips
from .data_funcs import create_dir_if_not_exist

from scipy import misc
import numpy as np
from osgeo import gdal
import os
import subprocess

class RasterCutter():  
    """Handles cutting of input raster"""
    
    def cut_raster_to_squares(self, input_raster, output_dir, prefix, tile_size):
        # https://gis.stackexchange.com/questions/14712/splitting-raster-into-smaller-chunks-using-gdal
        """TODO: Zero-pad outputs to fill extent"""
        dset = gdal.Open(input_raster)
        
        width = dset.RasterXSize
        height = dset.RasterYSize
        
        create_dir_if_not_exist(output_dir)

        for i in range(tile_size, width-tile_size, tile_size): # Cut out ragged edges 
            for j in range(tile_size, height-tile_size, tile_size):
                w = min(i+tile_size, width) - i
                h = min(j+tile_size, height) - j
                gdaltranString = "gdal_translate -of GTIFF -srcwin "+str(i)+", "+str(j)+", "+str(w)+", " \
                    +str(h)+" " + input_raster + " " + output_dir + prefix + str(i)+"_"+str(j)+".tif"
                subprocess.call(gdaltranString)        

    def _get_files_in_dir(self, directory):
        output_directory = self.output_directory
        files_in_dir = []
        
        for file in sorted(os.listdir(output_directory)):
            files_in_dir.append(os.path.join(output_directory, file))
        return files_in_dir


def copy_raster_band(target_raster, source_raster, out_file, band_to_copy, target_band):
    t_raster = gdal.Open(target_raster)
    s_raster = gdal.Open(source_raster)
    
    t_raster_copy = gdal.GetDriverByName('MEM').CreateCopy('', t_raster, 0)
    data = np.array(s_raster.GetRasterBand(band_to_copy).ReadAsArray())

    if t_raster_copy.GetRasterBand(target_band) == None:
        target_band = t_raster_copy.RasterCount+1
        t_raster_copy.AddBand()
    t_raster_copy.GetRasterBand(target_band).WriteArray(data)   
    
    t_raster_copy.FlushCache()
    gdal.GetDriverByName('GTiff').CreateCopy(out_file, t_raster_copy, 0)
    
    t_raster_copy = None
    t_raster = None
    s_raster = None

def dsm_to_byte_range(source, target, max_height):
    """corrects a DSM to be in range of Dutch values
    Minimum of -7, maximum of 500
    Max height of tile has to be specified. This is because GDAL reads np.max() line-by-line
    Bins the range from 0-255 depending on the maximum value of the raster
    Sources: https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
    and https://gis.stackexchange.com/questions/200464/how-to-replace-pixel-values-in-a-single-band-dem/200477
    Not a very clean and clear solution, could do with a refactor
    """
    temp = 'temp.tif'
    os.system("""gdal_calc.py -A {} --calc='((A>=-7)*(A<=500))*(A+7)+((A>500)*0)+((A<-7)*0)' --outfile={}""".format(source, temp))
    os.system("""gdal_calc.py -A {} --calc='A*(255/{})' --outfile={} --overwrite""".format(temp, max_height, temp))
    
    os.system("""gdal_translate -ot Byte {infile} {outfile}""".format(infile=temp, outfile=target))
    if os.path.isfile(temp): os.remove(temp)

def resample_raster(source, target, pixel_size):
    """For a given input raster and target, resamples the raster to the given pixel size"""
    os.system("""gdal_translate -a_nodata none -tr {ps} {ps} {infile} {outfile}""".format(infile=source, outfile=target, ps=pixel_size))


def get_raster_extent(raster_file):
    """Load a raster file and extract its bounding-box coordinates
    From https://gis.stackexchange.com/questions/57834/how-to-get-raster-corner-coordinates-using-python-gdal-bindings
    """
    raster = gdal.Open(raster_file)
    upperleft_x, xres, xskew, upperleft_y, yskew, yres  = raster.GetGeoTransform()
    lowerright_x = upperleft_x + (raster.RasterXSize * xres)
    lowerright_y = upperleft_y + (raster.RasterYSize * yres)
    raster = None
    return {"xmax":upperleft_x, "ymin":lowerright_y, "xmin":lowerright_x, "ymax":upperleft_y}


def clip_raster(src_path, target_path, bbox):
    """With a given bounding box, clips a raster to the given extent using GDAL translate.
    ! GDAL must be installed with command line capabilities !
    """
    # os.system("gdal_translate -projwin 52930 407904 53226 407719 test_data/lufo15_zierikzee.tif test_data/lufo15_cutout_ver2.tif")
    os.system("""gdal_translate -projwin {} {} {} {} {} {}""".format(bbox["xmax"],bbox["ymax"],bbox["xmin"],bbox["ymin"], src_path, target_path))
    print("""gdal_translate -projwin {} {} {} {} {} {}""".format(bbox["xmin"],bbox["ymin"],bbox["xmax"],bbox["ymax"], src_path, target_path))


def rasterize_polygon(input_vector_path, output_geotiff_path, pixel_size, out_dimensions=3):
    """Loads an OGR-compatible vector geometry file, then rasterizes inputs to a geotiff
    """
    out_dims = ("1 "*out_dimensions)[:-1]
    os.system("gdal_rasterize -burn {} -ot Byte -tr {ps} {ps} {} {}".format(out_dims, input_vector_path, output_geotiff_path, ps = pixel_size))
 
# =============================================================================
#     Next section is purposely not refactored as osgeo tools typically do not respond well to being
#     accessed in parts (e.g. returning an opened raster is prone to crashing),
#     while closing & re-opening the raster induces unnecessary read/write operations
# =============================================================================
    
def create_output_tile(image_array, raster_file, raster_out_filepath):
    """For a given greyscale class array, loads the matching source raster file, creates 
    a single geotiff tile, and writes the numpy array into the new raster"""
    driver = gdal.GetDriverByName('GTiff')
    
    raster = gdal.Open(raster_file)
    rows = raster.RasterYSize
    cols = raster.RasterXSize
    bands = image_array.shape[0]
    
    if bands == 1:
        datatype = gdal.GDT_Byte
    else:
        datatype = gdal.GDT_Int16
    
    raster_out = driver.Create(raster_out_filepath,cols,rows,bands,datatype)
    raster_out.SetGeoTransform(raster.GetGeoTransform())
    raster_out.SetProjection(raster.GetProjection())
    
    # If matrix has >3 dims (bands,width,height), write each band array in a separate new band
    if len(image_array.shape) > 2:
        for i, matrix in enumerate(image_array):
            out_band = i+1 # Must be explicitly defined beforehand else GDAL gets sassy
            raster_out.GetRasterBand(out_band).WriteArray(matrix)
    else:
        raster_out.GetRasterBand(1).WriteArray(image_array)
    
    raster_out.FlushCache()
    raster = None
    raster_out = None    

def mosaic_rasters(raster_tile_directory, raster_out_path, temp_file):
    """For a given directory with rasters, output path, and temporary text file path,
    mosaics the collection of tiles into a single raster"""
    all_rasters = i_manips.get_images(raster_tile_directory)
    _write_file_list(all_rasters, temp_file)
    
    os.system("gdal_merge.py -o {} -q -v --optfile {}".format(raster_out_path, temp_file))
    
    os.remove(temp_file)
    
def _write_file_list(file_list, temp_file='temp.txt'):
    """With a given input array, Writes a temporary file to use while merging"""
    with open(temp_file, 'w') as f:
        for _,file_location in enumerate(file_list):
            f.write(file_location+'\n')
            
if __name__ == '__main__':
#    source_raster = '/home/anteagroup/Documents/data/LUFO_zz/pdok_25cm_zz_nir/nir.tif'
#    target_raster = '/home/anteagroup/Documents/deeplearning/code/bag_project_p2/data/rasters/src/lufo15_zierikzee.tif'
#    out_path = '/home/anteagroup/Documents/deeplearning/code/bag_project_p2/data/rasters/src/lufo15_zierikzee_nir.tif'
#    copy_raster_band(out_path, source_raster, target_raster, 3, 4)
    
    source = '/home/anteagroup/Documents/data/LUFO_zz/r_64hz1.tif'
    target = '/home/anteagroup/Documents/data/LUFO_zz/ahn3_10cm.tif'
    resample_raster(source, target, 0.1)
    
