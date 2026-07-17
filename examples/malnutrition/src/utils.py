# ==================================================== #
# LIBRARIES
# ==================================================== #
import numpy as np
import pandas as pd
import math
from pyproj import Transformer, CRS
from shapely.geometry import Point
from shapely.ops import transform
from functools import partial


# ==================================================== #
# AREA OF INTEREST AROUND POSITION
# ==================================================== #
def get_bounding_box_coords(lat, lon, width_km):

    """
    Compute coordinates of the bounding box centered at (``lon``, ``lat``) with a width of ``width_km`` kilometers. Updated and modified version of [https://github.com/p4tr1ckc4rs0n/dhs-landsat/blob/main/src/utils.py](https://github.com/p4tr1ckc4rs0n/dhs-landsat/blob/main/src/utils.py).
    
    :param lat: latitude of the bounding box centroid.
    :type lat: float64
    :param lon: longitude of the bounding box centroid.
    :type lon: float64
    :param width_km: bounding box width.
    :type width_km: float64
    :return: min and max coordinates of the bounding box.
    :rtype: list
    """
    
    # define CRS (Coordinate Reference System)
    aeqd_crs = CRS(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0")
    wgs84_crs = CRS("+proj=longlat +datum=WGS84")

    # define transformation from aeq to wgs84
    transformer = Transformer.from_crs(aeqd_crs, wgs84_crs)
    project = partial(transformer.transform)

    # compute coordinates of rectangle area of interest
    buf = Point(lat, lon).buffer(0.5* width_km * 1000)
    aoi = transform(project, buf).exterior.envelope
    xmin, ymin, xmax, ymax = [round(coord, 4) for coord in aoi.bounds]

    return([xmin, ymin, xmax, ymax])

# get_bounding_box_coords(1.23,-5.2, 10)


# ==================================================== #
# GET COUNTRY CODES FROM COUNTRY NAMES
# ==================================================== #
def get_country_codes(country_names, df_countries, verbose=True):
    n_countries = len(country_names)
    country_codes = []
    for k in range(n_countries):
        country_name = country_names[k]
        country_code = df_countries.loc[df_countries["CountryName"] == country_name]["DHS_CountryCode"].values[0]
        country_codes.append(country_code)
        if verbose: print("[%s] %s" %(country_code, country_name))
    return(country_codes)


# ==================================================== #
# GET COUNTRY NAMES FROM COUNTRY CODES
# ==================================================== #
def get_country_names(country_codes, df_countries, verbose=True):
    n_countries = len(country_codes)
    country_names = []
    for k in range(n_countries):
        country_code = country_codes[k]
        country_name = df_countries.loc[df_countries["DHS_CountryCode"] == country_code]["CountryName"].values[0]
        country_names.append(country_name)
        if verbose: print("[%s] %s" %(country_code, country_name))
    return(country_names)


# ==================================================== #
# GET COUNTRY ISO2 FROM COUNTRY NAMES
# ==================================================== #
def get_country_iso2(country_names, df_countries, verbose=True):
    n_countries = len(country_names)
    country_isos = []
    for k in range(n_countries):
        country_name = country_names[k]
        country_iso = df_countries.loc[df_countries["CountryName"] == country_name]["ISO2_CountryCode"].values[0]
        country_isos.append(country_iso)
        if verbose: print("[%s] %s" %(country_iso, country_name))
    return(country_isos)


# ==================================================== #
# GET COUNTRY ISO3 FROM COUNTRY NAMES
# ==================================================== #
def get_country_iso3(country_names, df_countries, verbose=True):
    n_countries = len(country_names)
    country_isos = []
    for k in range(n_countries):
        country_name = country_names[k]
        country_iso = df_countries.loc[df_countries["CountryName"] == country_name]["ISO3_CountryCode"].values[0]
        country_isos.append(country_iso)
        if verbose: print("[%s] %s" %(country_iso, country_name))
    return(country_isos)


# # ==================================================== #
# # REQUEST DATA USING DHS API BY QUERIES
# # ==================================================== #
# # request data using dhs api by queries 
# def query_dhs_api(breakdown=[], countryIds=[], surveyYear=[], surveyIds=[], indicatorIds=[], characteristicLabel=[], returnGeometry=[]):

#     # define dictionary of requested queries
#     queries = {"breakdown":breakdown, "countryIds":countryIds, "surveyYear":surveyYear, "surveyIds":surveyIds, "indicatorIds":indicatorIds, "characteristicLabel":characteristicLabel, "returnGeometry":returnGeometry}

#     # build url to request api
#     api_url = "https://api.dhsprogram.com/rest/dhs/data?"
#     for query in queries.keys():
#         if len(queries[query]) > 0:
#             api_url = api_url + query + "=" + ",".join(queries[query]) + "&"
#     api_url = api_url[:-1]

#     # display info
#     print("requested url to DHS API : %s" % (api_url))

#     # request data through url api
#     req = urllib.request.urlopen(api_url)
#     if(req.status==200):
#         response = json.loads(req.read())
#     else:
#         print("ERROR - Request failed [status = %d]" % req.status)

#     # convert dictionary to dataframe
#     df = pd.DataFrame.from_dict(response["Data"])

#     return(df, api_url)

# # POSSIBLE QUERIES
# # breakdown           : "all", "national", "subnational", "cluster"
# # countryIds          : df_countries["DHS_CountryCode"].unique()
# # surveyYear          : df_surveys["SurveyYear"].unique()
# # surveyId            : df_surveys["SurveyId"].unique()
# # indicatorIds        : df_indicators["IndicatorId"].unique()
# # characteristicLabel : "Total", "Male", "Female", "Urban", "Rural", "48-59", etc.
# # returnGeometry      : "true", "false"
# # lang                : "en", "fr"
# # f                   : "html", "csv", etc.




# ==================================================== #
# ORTHODROMIC DISTANCE
# ==================================================== #
def ortho_distance(lon_1, lat_1, lon_2, lat_2):
    
    """
    Compute the orthodromic distance in kilometers between (lon_1, lat_1) and (lon_2, lat_2). 
    
    :param lon_1: longitude in degrees of the first position.
    :type lon_1: float
    :param lat_1: latitude in degrees of the first position.
    :type lat_1: float
    :param lon_2: longitude in degrees of the second position.
    :type lon_2: float
    :param lat_2: latitude in degrees of the second position.
    :type lat_2: float
    :return: the distance in kilometers between (lon_1, lat_1) and (lon_2, lat_2).
    :rtype: float
    
    Orthodromic distance is computed using the trigonometric haversine formula.
    """

    # convert degrees to radians
    lat_1 = math.pi/180*lat_1
    lat_2 = math.pi/180*lat_2
    lon_1 = math.pi/180*lon_1
    lon_2 = math.pi/180*lon_2

    # compute earth radius at mean latitude
    r_earth_equ = 6378.137
    r_earth_pol = 6356.752
    lat_mean = (lat_1 + lat_2)/2
    r_earth = np.sqrt(((r_earth_equ**2 * np.cos(lat_mean))**2 + (r_earth_pol**2 * np.sin(lat_mean))**2) / ((r_earth_equ * np.cos(lat_mean))**2 + (r_earth_pol * np.sin(lat_mean))**2))

    # longitude and latitude differences
    dlat = lat_2 - lat_1
    dlon = lon_2 - lon_1

    # compute great-circle distance using haversine formula
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(lat_1) * np.cos(lat_2) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    hav_dist = r_earth * c

    return(hav_dist)


# ==================================================== #
# COMPARE SETS
# ==================================================== #
def compare_sets(array_1, array_2, verbose=True):

    # define sets from array
    s1 = set(array_1)
    s2 = set(array_2)

    # compute info about sets
    s1_n_cs2 = s1.difference(s2)
    cs1_n_s2 = s2.difference(s1)
    s1_n_s2 = s1.intersection(s2)
    s1_u_s2 = s1.union(s2)

    # display infos
    if verbose: print("S1 = %d | S2 = %d\nS1 n CS2 = %d | CS1 n S2 = %d\nS1 n S2 = %d | S1 u S2 = %d" % (len(s1), len(s2), len(s1_n_cs2), len(cs1_n_s2), len(s1_n_s2), len(s1_u_s2)))

    return s1_n_cs2, cs1_n_s2, s1_n_s2, s1_u_s2


# ==================================================== #
# PRINT DF COLUMNS
# ==================================================== #
def show_all_columns(df):
    for column in df.columns:
        print(column)


# ==================================================== #
# DISPLAY PROGRESS
# ==================================================== #
def display_progress(k, n, freq=5):
    n_lz = int(math.log10(n))+1
    if (int(n/(100/freq))>0) and (k % int(n/(100/freq)) == 0): 
        print("[%s/%d] %.1f %%" % (str(k).zfill(n_lz), n, 100*k/n))


# ==================================================== #
# AGGREGATE INDICATOR BY COUNTRIES
# ==================================================== #
def aggregate_by_countries(data, column_name, method=["mean", "sum", "q50"]):

    # extract countries from dataset
    countries_of_interest = data["country_iso3"].unique()
    n_countries = len(countries_of_interest)

    data_agg = pd.DataFrame(countries_of_interest, columns=["country_iso3"])
    data_agg[column_name] = [0.0] * n_countries
    for country_iso3 in countries_of_interest:
        df = data.loc[data["country_iso3"]==country_iso3].reset_index(drop=True)
        if method == "mean":
            data_agg.loc[data_agg["country_iso3"]==country_iso3, column_name] = df[column_name].mean()
        if method == "sum":
            data_agg.loc[data_agg["country_iso3"]==country_iso3, column_name] = df[column_name].sum()
        if method == "q50":
            data_agg.loc[data_agg["country_iso3"]==country_iso3, column_name] = df[column_name].quantile(0.50)        

    return data_agg


# ==================================================== #
# DISCRETIZE CONTINUOUS ARRAY
# ==================================================== #
def array_discretizer(continous_array, boundaries, zero_category=True):
    n_boundaries = len(boundaries)
    discrete_array = np.array([0]*len(continous_array))
    for k in range(n_boundaries-1):
        lower_bound = boundaries[k]
        upper_bound = boundaries[k+1]
        discrete_array[(continous_array>lower_bound) & (continous_array<=upper_bound)] = k+1
    
    if(not zero_category):
        discrete_array[discrete_array==0] = 1
        discrete_array = discrete_array-1
    
    return(discrete_array)


# ==================================================== #
# DISCRETIZE CONTINUOUS DATAFRAME COLUMNS
# ==================================================== #
def df_discretizer(df, con_col, cat_col, boundaries, zero_category=True):
    df[cat_col] = array_discretizer(df[con_col], boundaries, zero_category)  

    return(df)


# ==================================================== #
# AVERAGE POOLING NUMPY ARRAY (C, H, W)
# ==================================================== #
# [https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy]
def average_pooling_array(array, factor):

    # get dimensions
    n_channels, height, width = array.shape

    # compute new dimensions
    height_reduced = height // factor
    width_reduced = width // factor

    # average pooling
    array = array[:, :height_reduced*factor, :width_reduced*factor]
    array = array.reshape(n_channels,height_reduced,factor,width_reduced,factor)
    array = array.mean(axis=(2,4))

    return array

# array = np.random.rand(1, 10*3, 10*3)
# array_pooled = average_pooling_array(array, 3)


# ================================================================================================ #
# NEAR-SQUARE GRID LAYOUT
# ================================================================================================ #
def nearsq_grid_layout(n):

    """
    Compute by brute force search the near-square grid layout dimensions.
    
    :param n: integer we want the near-square layout dimensions.
    :type n: int
    :return: the near-square layout dimensions (a,b) minimising |n**0.5-a| + |n**0.5-b| constraining n<= a*b and a<=b.
    :rtype: tuple(int, int)
            
    .. note::
        n>0.
    """

    # init
    best_score = 2*n
    ub = int(n**0.5)+1

    # loop over possible values
    for a in range(1, ub+1):
        b = math.ceil(n/a)
        score = abs(n**0.5-a) + abs(n**0.5-b)

        # update score and keep solution
        if (score < best_score):
            best_score, solution = score, (a, b)

    return solution

# # examples
# for n in range(1,100):
#     a, b = nearsq_grid_layout(n)
#     print("%d: %d x %d = %d" % (n, a, b, a*b))