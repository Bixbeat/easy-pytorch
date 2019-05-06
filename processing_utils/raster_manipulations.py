from . import image_manipulations as i_manips
from .data_funcs import create_dir_if_not_exist

from scipy import misc
import numpy as np
from osgeo import gdal
import ogr
import fiona
import subprocess
import os
from shutil import move

from os.path import dirname, basename, join

class RasterCutter():
    """
    Handles cutting of raster files to smaller chunks
    """
    def cut_raster_to_squares(self, input_raster, output_dir, prefix, tile_size):
        """
        Takes an input raster, then creates equal-sized tiles of tile_size pixels
        from across the image.
            :param input_raster: Original raster to tile
            :param output_dir: Directory to write output tiles to 
            :param prefix: Prefix to add to the tiles
            :param tile_size: Specified size in the units of the raster coordinate system
        """   # https://gis.stackexchange.com/questions/14712/splitting-raster-into-smaller-chunks-using-gdal
        """TODO: Zero-pad outputs to fill extent"""
        create_dir_if_not_exist(output_dir)
        dset = gdal.Open(input_raster)
        
        width = dset.RasterXSize
        height = dset.RasterYSize
        
        create_dir_if_not_exist(output_dir)

        for w in range(0, width, tile_size):
            for h in range(0, height, tile_size):

                # Scales image extent back to max width/height where applicable
                if w+tile_size > width:
                    w = width - tile_size
                if h+tile_size > height:                    
                    h = height - tile_size

                ts = str(tile_size)
                clip_single_raster(w, h, ts, input_raster, output_dir, prefix)

def raster_pts_to_tiles(input_pts, aerial_img, tile_size, output_dir, prefix, p_size_scaling):
    create_dir_if_not_exist(output_dir)
    dset = gdal.Open(aerial_img)
    extent = get_raster_extent(aerial_img)
    
    width = dset.RasterXSize
    height = dset.RasterYSize    

    with fiona.open(input_pts) as pts:
        for point in pts:
            left = int(point['geometry']['coordinates'][0])
            left = ((left - extent['xmax'])*p_size_scaling) - tile_size/2

            bottom = int(point['geometry']['coordinates'][1])
            bottom = ((extent['ymax'] - bottom)*p_size_scaling) - tile_size/2

            if left < 0:
                left = 0
            elif left + tile_size > width:
                left = width - tile_size

            if bottom < 0:
                bottom = 0
            elif bottom + tile_size > height:
                bottom = bottom - tile_size
            
            clip_single_raster(left, bottom, tile_size, aerial_img, output_dir, prefix)

def geojson_from_preds(file_name, preds,
                            layer_name="features",
                            field_name="features"):
    dset = gdal.Open(preds)
    band = dset.GetRasterBand(1)

    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(file_name)
    dst_layer = dst_ds.CreateLayer(layer_name, srs=None)

    fd = ogr.FieldDefn(field_name, ogr.OFTInteger)
    dst_layer.CreateField(fd)
    dst_field = 1

    gdal.Polygonize(band, None, dst_layer, dst_field, [], callback=None)

def clip_single_raster(w, h, ts, input_raster, out_dir, prefix):
    transl_str = "gdal_translate -of GTIFF -srcwin "+str(w)+", "+str(h)+", "+str(ts)+", " \
        +str(ts)+" " + input_raster + " " + out_dir + prefix + str(w)+"_"+str(h)+".tif"
    FNULL = open(os.devnull, 'w') # Get the subprocess to shut up
    subprocess.call(transl_str, shell=True, stdout=FNULL)        

def copy_raster_band(source_raster, target_raster, out_file, band_to_copy, target_band):
    """
    # Copies the raster band of an existing raster to another raster file,
    # then writes the output to a new file.
    #     :param target_raster: The raster filepath to copy from
    #     :param source_raster: Target raster to merge band into
    #     :param out_file: Output raster filepath
    #     :param band_to_copy: Band number of the source raster to copy
    #     :param target_band: Band number to copy the source band into
    """
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

def add_rel_lum(input_tile, out_name):
    out_dir = os.path.dirname(out_name)
    create_dir_if_not_exist(out_dir)

    lum_tile = gdal.Open(input_tile)
    lum_tile_array = np.array(lum_tile.ReadAsArray())
    rel_lum = 0.21626 * lum_tile_array[0] + 0.7152 * lum_tile_array[1] + 0.0722 * lum_tile_array[2]

    lum_raster_copy = gdal.GetDriverByName('MEM').CreateCopy('', lum_tile, 0)

    target_band = lum_raster_copy.RasterCount+1
    lum_raster_copy.AddBand()
    lum_raster_copy.GetRasterBand(target_band).WriteArray(rel_lum)

    gdal.GetDriverByName('GTiff').CreateCopy('temp', lum_raster_copy, 0)
    move('temp', out_name)

def resample_raster(source, target, pixel_size):
    """For a given input raster and target, resamples the raster to the given pixel size"""
    subprocess.call("""gdal_translate -a_nodata none -tr {ps} {ps} {infile} {outfile}""".format(infile=source, outfile=target, ps=pixel_size))

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
    # subprocess.call("gdal_translate -projwin 52930 407904 53226 407719 test_data/lufo15_zierikzee.tif test_data/lufo15_cutout_ver2.tif")
    subprocess.call(f"""gdal_translate -projwin {bbox["xmax"]} {bbox["ymax"]} {bbox["xmin"]} {bbox["ymin"]} {src_path} {target_path}""", shell=True)

def rasterize_polygon(input_vec_path, output_gtiff_path, p_size, val_color_pairs=None, nodata_value=-9999):
    """Loads an OGR-compatible vector geometry file, then rasterizes inputs to a geotiff
    """
    tiff_dir = dirname(output_gtiff_path)
    out_tiff = basename(output_gtiff_path)
    create_dir_if_not_exist(tiff_dir)

    # GDAL is awfully specific about subdirectories, so we construct the full path to save headaches
    script_dir = os.getcwd()
    vec_dir = dirname(input_vec_path)
    vec_name = basename(input_vec_path)
    full_vec_path = join(script_dir, vec_dir, vec_name)

    # If indiscriminately assigning all polys to one class, only burn a single value where poylgons are present
    if not val_color_pairs:
        rasterize_str = f"gdal_rasterize -burn 1 -ot Byte -tr {p_size} {p_size} {full_vec_path} {out_tiff}"
        subprocess.call(rasterize_str, shell=True)
    else:
        for i, pair in enumerate(val_color_pairs):
            # Writes a new raster at iteration 0, then burns onto this raster at every iteration
            if i == 0:
                rasterize_str = f'''gdal_rasterize -where {pair[0]} -burn {pair[1]} -ot Byte -tr {p_size} {p_size} {full_vec_path} {out_tiff}'''
            else:
                rasterize_str = f"gdal_rasterize -where {pair[0]} -burn {pair[1]} {full_vec_path} {out_tiff}"
            subprocess.call(rasterize_str, shell=True)
    
    os.rename(out_tiff, output_gtiff_path)
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
        datatype = gdal.GDT_Float32
    
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
    tiff_dir = dirname(raster_out_path)
    create_dir_if_not_exist(tiff_dir)

    all_rasters = i_manips.get_images(raster_tile_directory)
    _write_file_list(all_rasters, temp_file)
    
    subprocess.call("gdal_merge.py -ot float32 -o {} -q -v --optfile {}".format(raster_out_path, temp_file), shell=True)
    
    os.remove(temp_file)
    
def _write_file_list(file_list, temp_file='temp.txt'):
    """With a given input array, Writes a temporary file to use while merging"""
    with open(temp_file, 'w') as f:
        for _,file_location in enumerate(file_list):
            f.write(file_location+'\n')