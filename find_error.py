from __future__ import print_function

import gc
import logging
import math
import os
import subprocess
import sys
from math import ceil, floor

import fiona
import geopandas as gpd
import numpy as np
import osgeo
import pandas as pd
import pycrs
import rasterio
import rasterio.mask
import rasterio.rio
from fiona.crs import from_epsg

# import shapefile
from osgeo import gdal, ogr, osr
from pyproj import Proj
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.plot import show, show_hist
from shapely.geometry import box

import common

logging.getLogger().setLevel(logging.INFO)

##########################
# https://stackoverflow.com/questions/3662361/fill-in-missing-values-with-nearest-neighbour-in-python-numpy-masked-arrays

import sys

from scipy import ndimage as nd

SCRIPTS_DIR = os.path.join(os.path.dirname(sys.executable))
sys.path.append(SCRIPTS_DIR)
import osgeo_utils.gdal_merge as gm


def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output:
        Return a filled array.
    """
    # import numpy as np
    # import scipy.ndimage as nd

    if invalid is None:
        invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]


###########################

CRS_WGS84 = 4326
CRS_LAMBERT_ATLAS = 3978
KM_TO_M = 1000
BUFFER_KM = 100 * KM_TO_M

# CELL_SIZE = 100
CELL_SIZE = 250
CELL_SIZE_OUT = 100 * KM_TO_M
DATA_DIR = os.path.realpath("/appl/data")
EXTRACTED_DIR = os.path.join(DATA_DIR, "extracted")
DOWNLOAD_DIR = os.path.join(DATA_DIR, "download")
GENERATED_DIR = os.path.join(DATA_DIR, "error/generated")
INTERMEDIATE_DIR = os.path.join(DATA_DIR, "error/intermediate")
# put in the folder structure so that firestarr can reference grid/100m and
# have this be the default
# DIR = os.path.join(GENERATED_DIR, f"{CELL_SIZE_OUT}m")
DIR = GENERATED_DIR
TMP = os.path.realpath("/appl/data/tmp")
CREATION_OPTIONS = [
    "TILED=YES",
    "BLOCKXSIZE=256",
    "BLOCKYSIZE=256",
    "COMPRESS=DEFLATE",
    "PREDICTOR=2",
    "ZLEVEL=9",
]
CREATION_OPTIONS_FUEL = CREATION_OPTIONS + []
CREATION_OPTIONS_DEM = CREATION_OPTIONS + []
EARTHENV = os.path.join(DATA_DIR, "gis/input/elevation/EarthEnv.tif")
FUEL_RASTER = os.path.join(DATA_DIR, f"tom/FBP_{CELL_SIZE}m_ts_20241015.tif")
HEXELS = os.path.join(DATA_DIR, "tom/mask_hexels/mask_hexs_actual.shp")

INT_FUEL = os.path.join(INTERMEDIATE_DIR, "fuel")
DRIVER_SHP = ogr.GetDriverByName("ESRI Shapefile")
DRIVER_TIF = gdal.GetDriverByName("GTiff")
DRIVER_GDB = ogr.GetDriverByName("OpenFileGDB")

# FIX: seriously does not like uint for some reason
# DATATYPE_FUEL = gdal.GDT_Int16
DATATYPE_FUEL = gdal.GDT_UInt16
DATATYPE_DEM = gdal.GDT_Int16

NODATA_FUEL = 0
BUFFER_DEGREES = 0


def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json

    return [json.loads(gdf.to_json())["features"][0]["geometry"]]


if not os.path.exists(DIR):
    os.makedirs(DIR)
if not os.path.exists(TMP):
    os.makedirs(TMP)


def wkt_from_hexel(hex, meridian):
    # try this format because that's what the polygon rasterize process is generating, and they need to match
    wkt = f'PROJCS["NAD_1983_UTM_{hex}",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",{meridian}],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    return wkt, meridian


def make_hexel(hexel):
    logging.info("Hexel {}: Starting".format(hexel.hex))
    dem = clip_dem(EARTHENV, hexel)
    fbp = clip_fuel(FUEL_RASTER, hexel)
    logging.info("Hexel {}: Done".format(hexel.hex))
    gc.collect()


import multiprocessing
from multiprocessing import Pool

# if __name__ == "__main__":
#     if not os.path.exists(INT_FUEL):
#         os.makedirs(INT_FUEL)
#     df_hexels = gpd.read_file(HEXELS)
#     df_hexels.geometry = df_hexels.buffer(BUFFER_KM)
#     df_hexels = df_hexels.to_crs(CRS_WGS84)
#     df_hexels["meridian"] = df_hexels.centroid.x
#     df_hexels = pd.concat([df_hexels, df_hexels.bounds], axis=1)
#     del df_hexels["geometry"]
#     hexels = [r[1] for r in df_hexels.iterrows()]
#     # small limit due to amount of disk access
#     num_threads = int(min(len(hexels), multiprocessing.cpu_count() / 4))
#     p = Pool(num_threads)
#     results = p.map(make_hexel, hexels)
#     p.close()
#     p.join()
#     logging.info("Done")


# def check_error(fp, zone):
#     base_tif = os.path.join(INT_FUEL, "base_{}".format(zone).replace(".", "_")) + ".tif"
#     if not os.path.exists(base_tif):
#         wkt, meridian = wkt_from_zone(zone)
#         proj_srs = osr.SpatialReference(wkt=wkt)
#         toProj = Proj(proj_srs.ExportToProj4())
#         lat = (meridian, meridian)
#         lon = (common.BOUNDS["latitude"]["min"], common.BOUNDS["latitude"]["max"])
#         df = pd.DataFrame(np.c_[lat, lon], columns=["Longitude", "Latitude"])
#         x, y = toProj(df["Longitude"].values, df["Latitude"].values)
#         MIN_EASTING = 300000
#         MAX_EASTING = 700000
#         MIN_NORTHING = int(y[0] / 100000) * 100000
#         MAX_NORTHING = (int(y[1] / 100000) + 1) * 100000
#         ds = gdal.Open(fp)
#         out_image = None
#         out_transform = None
#         data = rasterio.open(fp)
#         srcWkt = data.crs.wkt
#         data.close()
#         srcSRS = osr.SpatialReference()
#         srcSRS.ImportFromWkt(data.crs.wkt)
#         dstSRS = osr.SpatialReference()
#         dstSRS.ImportFromWkt(wkt)
#         rb = ds.GetRasterBand(1)
#         no_data = rb.GetNoDataValue()
#         rb = None
#         ds = gdal.Warp(
#             base_tif,
#             ds,
#             format="GTiff",
#             outputBounds=[MIN_EASTING, MIN_NORTHING, MAX_EASTING, MAX_NORTHING],
#             outputType=DATATYPE_FUEL,
#             creationOptions=CREATION_OPTIONS_FUEL,
#             xRes=CELL_SIZE_OUT,
#             yRes=CELL_SIZE_OUT,
#             srcNodata=no_data,
#             dstNodata=no_data,
#             srcSRS=srcWkt,
#             dstSRS=wkt,
#         )
#         # band.SetNoDataValue(NODATA_FUEL)
#         ds = None
#         # fix_nodata(base_tif)
#         fix_nodata(base_tif, NODATA_FUEL)
#     ds = gdal.Open(base_tif, 1)
#     rows = ds.RasterYSize
#     cols = ds.RasterXSize
#     rb = ds.GetRasterBand(1)
#     no_data = rb.GetNoDataValue()
#     rb = None
#     ds = None
#     return base_tif, cols, rows, no_data


def fix_nodata(out_tif, no_data=None):
    # HACK: make sure nodata value is set because C code expects it even if nothing is nodata
    ds = gdal.Open(out_tif, 1)
    rb = ds.GetRasterBand(1)
    # always want a nodata value, so if one doesn't exist then make it
    if no_data is None:
        data_type = rb.DataType
        if gdal.GDT_UInt16 == data_type:
            no_data = int(math.pow(2, 16) - 1)
        elif gdal.GDT_Int16 == data_type:
            no_data = int(-math.pow(2, 15) - 1)
        else:
            raise RuntimeError(f"Unexpected data type when fixing no_data value: {data_type}")
        # HACK: if neither min or max is this value, then it must not be used?
        rb_min = rb.GetMinimum()
        rb_max = rb.GetMaximum()
        if not rb_min or not rb_max:
            (rb_min, rb_max) = rb.ComputeRasterMinMax(True)
        if no_data in [rb_min, rb_max]:
            raise RuntimeError(f"Could not set raster nodata value to {no_data} because it is already used")
    rb.SetNoDataValue(no_data)
    rb.FlushCache()
    rb = None
    ds = None


fp = FUEL_RASTER
# def check_error(fp=FUEL_RASTER):
# out_tif = os.path.join(DIR, f"error_{os.path.basename(fp)}")
out_tif = os.path.join(DIR, f"error_{CELL_SIZE_OUT}m.tif")
if not os.path.exists(out_tif):
    logging.info(f"Finding error for {fp}")
    ds = gdal.Open(fp)
    # out_image = None
    # out_transform = None
    # data = rasterio.open(fp)
    # srcWkt = data.crs.wkt
    # data.close()
    # srcSRS = osr.SpatialReference()
    # srcSRS.ImportFromWkt(data.crs.wkt)
    # dstSRS = osr.SpatialReference()
    # dstSRS.ImportFromWkt(wkt)
    rb = ds.GetRasterBand(1)
    no_data = rb.GetNoDataValue()
    rb = None
    # wkt, meridian = wkt_from_hexel(hexel.hex, hexel.meridian)
    # proj_srs = osr.SpatialReference(wkt=wkt)
    # toProj = Proj(proj_srs.ExportToProj4())
    # lon = (hexel.minx, hexel.maxx)
    # lat = (hexel.miny, hexel.maxy)
    # df = pd.DataFrame(np.c_[lon, lat], columns=["Longitude", "Latitude"])
    # x, y = toProj(df["Longitude"].values, df["Latitude"].values)
    # MIN_EASTING = int(x[0] / 100000) * 100000
    # MAX_EASTING = (int(x[1] / 100000) + 1) * 100000
    # MIN_NORTHING = int(y[0] / 100000) * 100000
    # MAX_NORTHING = (int(y[1] / 100000) + 1) * 100000
    ds = gdal.Warp(
        out_tif,
        ds,
        format="GTiff",
        creationOptions=CREATION_OPTIONS,
        xRes=CELL_SIZE_OUT,
        yRes=CELL_SIZE_OUT,
    )
    ds = None
    # fix_nodata(out_tif)
    # HACK: firestarr expects a nodata value, even if we don't have any nodata in the raster
    fix_nodata(out_tif, no_data)
    gc.collect()
    ds = gdal.Open(out_tif, 1)
    rb = ds.GetRasterBand(1)
    # always want a nodata value, so if one doesn't exist then make it
    if no_data is None:
        data_type = rb.DataType
        if gdal.GDT_UInt16 == data_type:
            no_data = int(math.pow(2, 16) - 1)
        elif gdal.GDT_Int16 == data_type:
            no_data = int(-math.pow(2, 15) - 1)
        else:
            raise RuntimeError(f"Unexpected data type when fixing no_data value: {data_type}")
        # HACK: if neither min or max is this value, then it must not be used?
        rb_min = rb.GetMinimum()
        rb_max = rb.GetMaximum()
        if not rb_min or not rb_max:
            (rb_min, rb_max) = rb.ComputeRasterMinMax(True)
        if no_data in [rb_min, rb_max]:
            raise RuntimeError(f"Could not set raster nodata value to {no_data} because it is already used")
    rb.SetNoDataValue(no_data)
    rb.FlushCache()
    rb = None
    ds = None
    # return out_tif


import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from pyproj import Transformer
from shapely.geometry import Point


def to_gdf(df, crs=CRS_WGS84):
    geometry = df["geometry"] if "geometry" in df else gpd.points_from_xy(df["lon"], df["lat"], crs=crs)
    return gpd.GeoDataFrame(df, crs=crs, geometry=geometry)


gc.collect()


def xy_to_latlong(x, y):
    # row, column
    xs, ys = rasterio.transform.xy(src.transform, y, x)
    lon, lat = transformer.transform(np.array(xs), np.array(ys))
    return lon, lat


file_out = out_tif.replace(".tif", ".parquet")
# file_out = out_tif.replace(".tif", ".gpkg")
if not os.path.exists(file_out):
    # need to look at resampled raster, not original
    with rasterio.open(out_tif) as src:
        band1 = src.read(1)
        height = band1.shape[0]
        width = band1.shape[1]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        crs = f"epsg:{CRS_WGS84}"
        transformer = Transformer.from_crs(src.crs, crs, always_xy=True)
        lons, lats = transformer.transform(np.array(xs), np.array(ys))
        df = pd.DataFrame(
            data={
                "lon": lons,
                "lat": lats,
                "x": cols.flatten(),
                "y": rows.flatten(),
            }
        )
        gc.collect()
        gdf = to_gdf(df, crs)
        df = None
        gc.collect()
        # gdf.to_file(file_out, driver="GPKG")
        gdf.to_parquet(file_out)
        # coords = gpd.GeoSeries(list(zip(lons.flatten(), lats.flatten())).map(Point), crs=crs)
        # points = coords.map(Point)

        # # use the feature loop in case shp is multipolygon
        # geoms = points.values
        # features = [i for i in range(len(geoms))]

        # out = gpd.GeoDataFrame(
        #     {'feature': features, 'geometry': geoms}, crs=src.crs)
        # out.to_file("pixel_center_points.shp")
# gdf = gpd.read_file(file_out)
gdf = gpd.read_parquet(file_out)

gdf["lon_int"] = gdf["lon"].astype(int)

from geographiclib.geodesic import Geodesic


def point_from_offset(p1, offset_x, offset_y):
    lon1, lat1 = xy_to_latlong(p1.x, p1.y)
    assert lon1 == p1.lon
    assert lat1 == p1.lat
    lon2, lat2 = xy_to_latlong(p1.x + offset_x, p1.y + offset_y)
    return lon2, lat2


def find_inverse(p1, offset_x, offset_y):
    lon2, lat2 = point_from_offset(p1, offset_x, offset_y)
    inv = Geodesic.WGS84.Inverse(p1.lat, p1.lon, lat2, lon2)
    return inv


# def find_distance(p1, offset_x, offset_y):
#     return find_inverse(p1, offset_x, offset_y)["s12"]


# def find_bearing(p1, offset_x, offset_y):
#     return find_inverse(p1, offset_x, offset_y)["a12"]


def find_bearing_distance(p1, offset_x, offset_y):
    inv = find_inverse(p1, offset_x, offset_y)
    return inv["azi1"], inv["s12"]


gdf_orig = gdf


# reset to gdf_orig to make copy/paste easier
gdf = gdf_orig

# NOTE: this is way more duplication of effort, but lets us get cells outside raster along edges without doing anything differently
# (0, 0) cell is top left corner
gdf[["bearing_down", "distance_down"]] = gdf.apply(
    lambda p1: find_bearing_distance(p1, 0, 1),
    axis=1,
    result_type="expand",
)
gdf[["bearing_up", "distance_up"]] = gdf.apply(
    lambda p1: find_bearing_distance(p1, 0, -1),
    axis=1,
    result_type="expand",
)
gdf[["bearing_left", "distance_left"]] = gdf.apply(
    lambda p1: find_bearing_distance(p1, -1, 0),
    axis=1,
    result_type="expand",
)
gdf[["bearing_right", "distance_right"]] = gdf.apply(
    lambda p1: find_bearing_distance(p1, 1, 0),
    axis=1,
    result_type="expand",
)

gdf.to_parquet(out_tif.replace(".tif", ".parquet"))
