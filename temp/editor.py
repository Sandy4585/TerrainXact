from flask import Flask, render_template, request, send_file
import subprocess
import os
from osgeo import gdal, ogr, osr
import zipfile
import tempfile
import io
import numpy as np
import logging
import ezdxf
import csv
from scipy.spatial import Delaunay

app = Flask(__name__, static_folder='static', static_url_path='/static')

logging.basicConfig(level=logging.DEBUG)

M_TO_FT = 3.28084  # Conversion factor from meters to feet

def create_temp_dir():
    temp_dir = os.path.join(tempfile.gettempdir(), 'terrain_processing_temp')
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def get_first_word(filename):
    return os.path.splitext(os.path.basename(filename))[0].split('_')[0]

def clip_raster(dem_path, kml_path):
    logging.debug("Clipping the raster with dem_path: %s and kml_path: %s", dem_path, kml_path)
    tmp_dir = create_temp_dir()
    tmp_output_path = os.path.join(tmp_dir, f'{get_first_word(kml_path)}_clipped_dem.tif')
    subprocess.run(['gdalwarp', '-cutline', kml_path, '-crop_to_cutline', dem_path, tmp_output_path], check=True)
    with open(tmp_output_path, 'rb') as f:
        clipped_data = f.read()
    logging.debug("Clipped raster data length: %d bytes", len(clipped_data))
    return clipped_data, tmp_dir

def transform_to_feet(clipped_dem_data, tmp_dir, kml_name):
    logging.debug("Transforming clipped DEM data to feet")
    tmp_input_path = os.path.join(tmp_dir, f'{get_first_word(kml_name)}_clipped_dem.tif')
    with open(tmp_input_path, 'wb') as tmp_input:
        tmp_input.write(clipped_dem_data)
    input_ds = gdal.Open(tmp_input_path, gdal.GA_Update)
    dem_band = input_ds.GetRasterBand(1)
    dem_array = dem_band.ReadAsArray()
    
    # Mask the no-data values
    nodata_value = dem_band.GetNoDataValue()
    dem_array = np.ma.masked_equal(dem_array, nodata_value)
    logging.debug("Original DEM array stats - min: %f, max: %f", dem_array.min(), dem_array.max())
    
    dem_array_feet = dem_array * M_TO_FT  # Conversion factor from meters to feet
    logging.debug("Converted DEM array stats - min: %f, max: %f", dem_array_feet.min(), dem_array_feet.max())
    
    # Unmask the no-data values
    dem_array_feet = dem_array_feet.filled(nodata_value)
    
    tmp_output_path = os.path.join(tmp_dir, f'{get_first_word(kml_name)}_clipped_dem(feet).tif')
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(tmp_output_path, input_ds.RasterXSize, input_ds.RasterYSize, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(input_ds.GetGeoTransform())
    out_ds.SetProjection(input_ds.GetProjection())
    
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(dem_array_feet)
    out_band.SetNoDataValue(nodata_value)
    
    out_ds.FlushCache()
    out_ds = None  # Close the output dataset
    
    with open(tmp_output_path, 'rb') as f:
        transformed_data = f.read()
    logging.debug("Transformed DEM data length: %d bytes", len(transformed_data))
    return transformed_data, tmp_dir

def generate_contours(clipped_dem_feet_data, tmp_dir, kml_name, interval=1):
    tmp_input_path = os.path.join(tmp_dir, f'{get_first_word(kml_name)}_clipped_dem(feet).tif')
    input_ds = gdal.Open(tmp_input_path, gdal.GA_Update)
    raster_band = input_ds.GetRasterBand(1)
    proj = osr.SpatialReference(wkt=input_ds.GetProjection())
    dem_nan = raster_band.GetNoDataValue()

    tmp_output_path = os.path.join(tmp_dir, f'{get_first_word(kml_name)}_shapefile.shp')
    contour_ds = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(tmp_output_path)
    contour_shp = contour_ds.CreateLayer('contour', proj, geom_type=ogr.wkbLineString25D)
    field_def = ogr.FieldDefn("ID", ogr.OFTInteger)
    contour_shp.CreateField(field_def)
    field_def = ogr.FieldDefn("elev", ogr.OFTReal)
    contour_shp.CreateField(field_def)

    gdal.ContourGenerate(raster_band, interval, 0, [], 1, dem_nan, contour_shp, 0, 1)
    contour_ds = None  # Close the contour dataset

    with open(tmp_output_path, 'rb') as f:
        contour_data = f.read()
    logging.debug("Generated contour data length: %d bytes", len(contour_data))
    return contour_data, tmp_dir

def convert_shapefile_to_dxf(shapefile_data, tmp_dir, kml_name):
    tmp_input_path = os.path.join(tmp_dir, f'{get_first_word(kml_name)}_shapefile.shp')
    tmp_output_path = os.path.join(tmp_dir, f'{get_first_word(kml_name)}_contours(feet).dxf')
    subprocess.run(['ogr2ogr', '-f', 'DXF', '-zfield', 'elev', tmp_output_path, tmp_input_path], check=True)
    with open(tmp_output_path, 'rb') as f:
        dxf_data = f.read()
    logging.debug("Converted DXF data length: %d bytes", len(dxf_data))
    return dxf_data, tmp_dir

def raster_to_points(clipped_dem_data, tmp_dir, kml_name):
    tmp_input_path = os.path.join(tmp_dir, f'{get_first_word(kml_name)}_clipped_dem.tif')
    input_ds = gdal.Open(tmp_input_path, gdal.GA_Update)
    band = input_ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    gt = input_ds.GetGeoTransform()

    tmp_output_path = os.path.join(tmp_dir, f'{get_first_word(kml_name)}_pvsyst_input(meters).csv')
    with open(tmp_output_path, 'w', newline='') as tmp_output:
        tmp_output.write("X,Y,Z\n")
        for y in range(band.YSize):
            for x in range(band.XSize):
                value = band.ReadAsArray(x, y, 1, 1)[0][0]
                if value != nodata:
                    px = gt[0] + x * gt[1] + y * gt[2]
                    py = gt[3] + x * gt[4] + y * gt[5]
                    tmp_output.write(f"{px},{py},{value}\n")

    with open(tmp_output_path, 'rb') as f:
        points_data = f.read()
    logging.debug("Generated points data length: %d bytes", len(points_data))
    return points_data, tmp_output_path, tmp_dir

# New functions for DXF generation with mesh and color
def read_csv(file_path):
    points_meters = []
    points_feet = []
    with open(file_path, newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            x = float(row['X'])
            y = float(row['Y'])
            z_meters = float(row['Z'])
            z_feet = z_meters * M_TO_FT
            points_meters.append((x, y, z_meters))
            points_feet.append((x, y, z_feet))
    return points_meters, points_feet

def create_dxf(points, output_file):
    doc = ezdxf.new(dxfversion='R2010')
    modelspace = doc.modelspace()
    for point in points:
        modelspace.add_point(point, dxfattribs={'layer': '3D Points'})
    doc.saveas(output_file)

def create_mesh(points):
    coords = np.array([(x, y) for x, y, z in points])
    tri = Delaunay(coords)
    return tri.simplices

def calculate_slope(p1, p2, p3):
    a = np.linalg.norm(np.array(p2) - np.array(p1))
    b = np.linalg.norm(np.array(p3) - np.array(p2))
    c = np.linalg.norm(np.array(p1) - np.array(p3))
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    height = 2 * area / a
    slope = np.degrees(np.arctan(height / a))
    return slope

def slope_to_color(slope):
    if slope > 30:
        return 1  # Red
    elif slope > 25:
        return 2  # Yellow
    elif slope > 20:
        return 3  # Green
    elif slope > 15:
        return 4  # Cyan
    elif slope > 10:
        return 5  # Blue
    elif slope > 5:
        return 6  # Magenta
    else:
        return 7  # White

def create_dxf_mesh(points, simplices, output_file):
    doc = ezdxf.new(dxfversion='R2010')
    modelspace = doc.modelspace()
    for simplex in simplices:
        pts = [points[i] for i in simplex]
        slope = calculate_slope(pts[0], pts[1], pts[2])
        color = slope_to_color(slope)
        
        face = modelspace.add_3dface([
            (pts[0][0], pts[0][1], pts[0][2]),
            (pts[1][0], pts[1][1], pts[1][2]),
            (pts[2][0], pts[2][1], pts[2][2]),
            (pts[2][0], pts[2][1], pts[2][2])  # Duplicate the last point for triangles
        ], dxfattribs={'layer': '3D Mesh', 'color': color})
        face.dxf.invisible_edges = 1 + 2 + 4  # Make all edges invisible
    doc.saveas(output_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    logging.debug("Upload route accessed")
    # Get uploaded files
    dem_file = request.files['dem_file']
    kml_file = request.files['kml_file']

    # Save uploaded files to a temporary directory
    temp_dir = create_temp_dir()
    dem_path = os.path.join(temp_dir, dem_file.filename)
    kml_path = os.path.join(temp_dir, kml_file.filename)
    dem_file.save(dem_path)
    kml_file.save(kml_path)

    # Process the uploaded files
    try:
        clipped_dem_data, tmp_dir = clip_raster(dem_path, kml_path)
        clipped_dem_feet_data, tmp_dir = transform_to_feet(clipped_dem_data, tmp_dir, kml_file.filename)
        contour_shp_data, tmp_dir = generate_contours(clipped_dem_feet_data, tmp_dir, kml_file.filename)
        dxf_data, tmp_dir = convert_shapefile_to_dxf(contour_shp_data, tmp_dir, kml_file.filename)
        points_data, csv_path, tmp_dir = raster_to_points(clipped_dem_data, tmp_dir, kml_file.filename)

        points_meters, points_feet = read_csv(csv_path)
        dxf_meters_path = os.path.join(tmp_dir, f"{get_first_word(kml_file.filename)}_3D_points(meters).dxf")
        dxf_feet_path = os.path.join(tmp_dir, f"{get_first_word(kml_file.filename)}_3D_points(feet).dxf")
        dxf_feet_mesh_path = os.path.join(tmp_dir, f"{get_first_word(kml_file.filename)}_Generated_Mesh(feet).dxf")

        create_dxf(points_meters, dxf_meters_path)
        create_dxf(points_feet, dxf_feet_path)
        simplices = create_mesh(points_feet)
        create_dxf_mesh(points_feet, simplices, dxf_feet_mesh_path)

        # Read the generated DXF files for zipping
        with open(dxf_meters_path, 'rb') as f:
            dxf_meters_data = f.read()
        with open(dxf_feet_path, 'rb') as f:
            dxf_feet_data = f.read()
        with open(dxf_feet_mesh_path, 'rb') as f:
            dxf_feet_mesh_data = f.read()

        # Create an in-memory zip file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zipf:
            logging.debug("Adding clipped_dem.tif to zip")
            zipf.writestr(f"{get_first_word(kml_file.filename)}_clipped_dem.tif", clipped_dem_data)
            logging.debug("Adding clipped_dem_feet.tif to zip")
            zipf.writestr(f"{get_first_word(kml_file.filename)}_clipped_dem(feet).tif", clipped_dem_feet_data)
            logging.debug("Adding contours-shape-file.shp to zip")
            zipf.writestr(f"{get_first_word(kml_file.filename)}_shapefile.shp", contour_shp_data)
            logging.debug("Adding contours.dxf to zip")
            zipf.writestr(f"{get_first_word(kml_file.filename)}_contours(feet).dxf", dxf_data)
            logging.debug("Adding pvsyst_shading_file.csv to zip")
            zipf.writestr(f"{get_first_word(kml_file.filename)}_pvsyst_input(meters).csv", points_data)
            logging.debug("Adding output_meters.dxf to zip")
            zipf.writestr(f"{get_first_word(kml_file.filename)}_3D_points(meters).dxf", dxf_meters_data)
            logging.debug("Adding output_feet.dxf to zip")
            zipf.writestr(f"{get_first_word(kml_file.filename)}_3D_points(feet).dxf", dxf_feet_data)
            logging.debug("Adding output_feet_mesh.dxf to zip")
            zipf.writestr(f"{get_first_word(kml_file.filename)}_Generated_Mesh(feet).dxf", dxf_feet_mesh_data)
        zip_buffer.seek(0)

        # Clean up temporary files
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(temp_dir)

        logging.debug("Returning the generated zip file")
        kml_file_name = os.path.splitext(kml_file.filename)[0]
        return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name=f'{kml_file_name}.zip')

    except Exception as e:
        logging.error("Error in upload route: %s", e)
        return "An error occurred during processing", 500

if __name__ == "__main__":
    app.run(debug=True)

