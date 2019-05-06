from owslib.wfs import WebFeatureService
from osgeo import ogr
import os

# http://geopython.github.io/OWSLib/
def download_bag_buildings(bbox, file_dir):
    """Connects to the Dutch geodata registry and retrieves all buildings of the BAG.
        // Deprecated, WFS does not supply all buildings
    """
    for i in range(0,100):
        wfs = WebFeatureService(url='https://geodata.nationaalgeoregister.nl/bag/wfs?request=GetCapabilities', version='2.0.0')
        response = wfs.getfeature(typename='bag:pand', bbox=bbox, startindex=i*1000) # Max features flag does not work
        filename = file_dir+"temp_bag{}.gml".format(i)
        out = open(filename, 'wb')
        out.write(bytes(response.read(), 'UTF-8'))
        out.close()
        
        # Check size of response, breaks once no response is given
        # Appended to prevent locking the memory address using os.seek_end, else the file cannot be written
        if response.seek(0, os.SEEK_END) < 1000: 
            os.remove(filename)
            break

def merge_ogr_layers(directory, out_filepath):
    """TODO: Does not work yet, fix merge""" 
    out_driver = ogr.GetDriverByName( 'SHP' )
    if os.path.exists(out_filepath):
        out_driver.DeleteDataSource(out_filepath)
    out_ds = out_driver.CreateDataSource(out_filepath)
    out_layer = out_ds.CreateLayer(out_filepath)
    out_ds = None
    
    fileList = os.listdir(directory)
    
    for file in fileList:
        if file[-4:] == ".gml":        
            dataset = ogr.Open(directory+file)
            lyr = dataset.GetLayer()
            for feat in lyr:
                out_feat = ogr.Feature(out_layer.GetLayerDefn())
                out_feat.SetGeometry(feat.GetGeometryRef().Clone())
                out_layer.CreateFeature(out_feat)
                out_feat = None
                out_layer.SyncToDisk()
    
if __name__ == '__main__':
    # temp_bag_gml_directory = './data/temp_gml/'
    # wfs_response = download_bag_buildings(bbox=(52603,407160,54730,408445), file_dir = temp_bag_gml_directory) # BAG WFS is kapuut
    
    """Hardcoded because OGR sucks"""
    # directory = 'd:/data/temp/'
    # out_filepath = 'd:/data/merged_bag.gml'    
    # merge_ogr_layers(directory, out_filepath)
    
    