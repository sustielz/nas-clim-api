import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import xarray as xr
import numpy as np
from functools import lru_cache
from data_sets import DATASETS
import pandas as pd



app = FastAPI(openapi_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
    max_age=600,
)

USE_S3 = os.environ.get("USE_S3", "false").lower() == "true"

# Open Zarr 
@lru_cache(maxsize=10)
def open_zarr(zarr_name: str) -> xr.Dataset:
    if USE_S3:
        import s3fs
        S3_BUCKET = os.environ.get("S3_BUCKET", "my-climate-data-2024")
        S3_REGION = os.environ.get("S3_REGION", "ap-northeast-1")
        fs    = s3fs.S3FileSystem(
            anon=False,
            client_kwargs={"region_name": S3_REGION}
        )
        store = s3fs.S3Map(root=f"{S3_BUCKET}/data/{zarr_name}.zarr", s3=fs)
        return xr.open_dataset(store, engine="zarr")
    else:
        return xr.open_dataset(f"data/{zarr_name}.zarr", engine="zarr")

# check api status
@app.get("/health")
async def health():
    return {"status": "ok"}

# list all data
@app.get("/datasets")
async def get_datasets():
    result = {}
    for key, ds in DATASETS.items():
        result[key] = {
            "label":        ds["label"],
            "available":    ds["zarr_file"] is not None,
            "has_time":     ds["has_time"],
            "variables":    ds["variables"],
            "color_group1": ds["color_group1"],
            "color_group2": ds["color_group2"],
            "ts_style":     ds["ts_style"],
        }
    return result

# check point
@app.get("/query/{dataset_id}/point")
async def query_point(
    dataset_id: str,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    if dataset_id not in DATASETS:
        raise HTTPException(404, "Dataset not found")

    ds_config = DATASETS[dataset_id]

    if ds_config["zarr_file"] is None:
        raise HTTPException(503, f"{ds_config['label']} data not available yet")

    if ds_config["has_time"]:
        raise HTTPException(400, "This dataset has time dimension")

    ds     = open_zarr(ds_config["zarr_file"])
    result = {}

    for var in ds_config["variables"]:
        val = float(
            ds[var].sel(lat=lat, lon=lon, method="nearest").values
        )
        # check nan and -999
        if np.isnan(val) or np.isinf(val) or val <= -999:
            result[var] = None
        else:
            result[var] = round(val, 4)

    return {
        "mode":      "point",
        "dataset":   dataset_id,
        "label":     ds_config["label"],
        "lat":       lat,
        "lon":       lon,
        "values":    result,
        "colors_g1": ds_config["color_group1"],
        "colors_g2": ds_config["color_group2"],
    }
    

# if time period
@app.get("/query/{dataset_id}/year")
async def query_year(
    dataset_id: str,
    lat:  float = Query(..., ge=-90, le=90),
    lon:  float = Query(..., ge=-180, le=180),
    year: int   = Query(...),
):
    if dataset_id not in DATASETS:
        raise HTTPException(404, "Dataset not found")

    ds_config = DATASETS[dataset_id]

    if ds_config["zarr_file"] is None:
        raise HTTPException(503, f"{ds_config['label']} data not available yet")

    ds     = open_zarr(ds_config["zarr_file"])
    result = {}

    for var in ds_config["variables"]:
        val = float(
            ds[var].sel(lat=lat, lon=lon, method="nearest")
                   .sel(time=year, method="nearest").values
        )
        result[var] = round(val, 4)

    return {
        "mode":      "single_year",
        "dataset":   dataset_id,
        "label":     ds_config["label"],
        "lat":       lat,
        "lon":       lon,
        "year":      year,
        "values":    result,
        "colors_g1": ds_config["color_group1"],
        "colors_g2": ds_config["color_group2"],
    }

# if yes time series
@app.get("/query/{dataset_id}/timeseries")
async def query_timeseries(
    dataset_id:  str,
    lat:         float = Query(..., ge=-90, le=90),
    lon:         float = Query(..., ge=-180, le=180),
    start_year:  int   = Query(...),
    end_year:    int   = Query(...),
):
    if dataset_id not in DATASETS:
        raise HTTPException(404, "Dataset not found")

    if start_year >= end_year:
        raise HTTPException(400, "start_year must be less than end_year")

    ds_config = DATASETS[dataset_id]

    if ds_config["zarr_file"] is None:
        raise HTTPException(503, f"{ds_config['label']} data not available yet")

    ds     = open_zarr(ds_config["zarr_file"])
    result = {}

    for var in ds_config["variables"]:
        data = (
            ds[var].sel(lat=lat, lon=lon, method="nearest")
                   .sel(time=slice(start_year, end_year))
        )
        result[var] = {
            "times":  [int(t) for t in data["time"].values],
            "values": [round(float(v), 4) for v in data.values],
            "min":    round(float(data.min()), 4),
            "max":    round(float(data.max()), 4),
        }

    return {
        "mode":       "timeseries",
        "dataset":    dataset_id,
        "label":      ds_config["label"],
        "lat":        lat,
        "lon":        lon,
        "start_year": start_year,
        "end_year":   end_year,
        "ts_style":   ds_config["ts_style"],
        "variables":  result,
        "colors":     ds_config["color_group1"] + ds_config["color_group2"],
    }

# Find which tanks are responsible for spill in a certain area
@app.get("/query/{dataset_id}/whichtanks")
async def query_grid(
    dataset_id: str, 
    lat:        float = Query(..., ge=-90, le=90),
    lon:        float = Query(..., ge=-180, le=180),   
    rad:        float = Query(0.001),
):

    ## Sanitize input
    if dataset_id not in ['s100yr']: raise HTTPException(503, "Data does not exist")
    if _spill_cache is None: raise HTTPException(503, "Marker positions not loaded yet")

    print('QUERYING SPILL CACHE')
    df = _spill_cache[ _spill_cache.lon.apply(lambda x: np.abs(x-lon) < rad) &   
                       _spill_cache.lat.apply(lambda y: np.abs(y-lat) < rad) ]
    print(df)
    print(df.ast_id)



    return {'ids': [int(i) for i in df.ast_id if not np.isnan(i)]}
#    return {
#        "lons":     df.lon,
#        "lats":     df.lat,
#        "ast_ids":  df.ast_id
#    }


## NOTE: This function currently does nothing since tanks are currently only queried one-at-a-time by frontent. Revisit.
# Fragility AST location
@app.get("/fragility/points")
async def get_fragility_points():
    if _ast_points_cache is None:
        raise HTTPException(503, "Fragility data not ready yet")
    return _ast_points_cache


#  Fragility clicking
@app.get("/fragility/ast/{ast_id}")
async def get_ast_detail(ast_id: int):
    if _ast_points_cache is None:
        raise HTTPException(503, "Fragility data not ready yet")

    # try cache , saving time
    pt = next(
        (p for p in _ast_points_cache["points"] if p["ast_id"] == ast_id),
        None
    )

    if pt is None:
        raise HTTPException(404, f"AST_ID {ast_id} not found")

    return {
        "ast_id": ast_id,
        "lat":    pt["lat"],
        "lon":    pt["lon"],
        "type":   pt["type"],
        "height": pt["height"],
        "pf": {
            "mean": pt["pf_mean"],
            "std":  pt["pf_std"],
        },
        "sv": {
            "mean": pt["sv_mean"],
            "std":  pt["sv_std"],
        },
        "flood25":  pt["flood25"],
        "flood50":  pt["flood50"],
        "flood100": pt["flood100"],
        "surge25":  pt["surge25"],
        "surge50":  pt["surge50"],
        "surge100": pt["surge100"],
        "wind25":  pt["mwspd25"],
        "wind50":  pt["mwspd50"],
        "wind100": pt["mwspd100"],
        
    }


# default ast
_ast_points_cache = None
def build_ast_cache():
    global _ast_points_cache
    try:
        ds1 = pd.read_pickle("data/100yr_fragility.pkl")
        ds2 = open_zarr("return_tank_levels").to_dataframe()
        ds = pd.merge(ds1, ds2)
        
        lats    = ds.Latitude.values
        lons    = ds.Longitude.values
        ids     = ds.index.values
        types   = ds.Type.values
        heights = ds.Height.values


        pf_mean = ds.mean_fail_prob.values 
        pf_std  = ds.std_fail_prob.values
        sv_mean = ds.mean_expected_vol.values
        sv_std  = ds.std_expected_vol.values 

        flood25 = ds.flood25 .values
        flood50 = ds.flood50 .values
        flood100= ds.flood100.values 
        surge25 = ds.surge25 .values
        surge50 = ds.surge50 .values
        surge100= ds.surge100.values
        mwspd25 = ds.mwspd25 .values
        mwspd50 = ds.mwspd50 .values
        mwspd100= ds.mwspd100.values





        
        points = []
        for i in range(len(ids)):
            points.append({
                "ast_id":  int(ids[i]),
                "lat":     round(float(lats[i]), 6),
                "lon":     round(float(lons[i]), 6),
                "type":    str(types[i]),
                "height":  round(float(heights[i]), 2),
                "pf_mean": [round(float(val), 6) for val in pf_mean[i]],
                "pf_std":  [round(float(val), 6) for val in pf_std[i]],
                "sv_mean": [round(float(val), 4) for val in sv_mean[i]],
                "sv_std":  [round(float(val), 4) for val in sv_std[i]],
                "flood25":   round(float(flood25 [i]),  4),
                "flood50":   round(float(flood50 [i]),  4),
                "flood100":  round(float(flood100[i]),  4),
                "surge25":   round(float(surge25 [i]),  4),
                "surge50":   round(float(surge50 [i]),  4),
                "surge100":  round(float(surge100[i]),  4),
                "mwspd25":   round(float(mwspd25 [i]),  4),
                "mwspd50":   round(float(mwspd50 [i]),  4),
                "mwspd100":  round(float(mwspd100[i]),  4),
            })

        _ast_points_cache = {"points": points, "count": len(points)}
        print(f"✓ AST cache built: {len(points)} points")

    except Exception as e:
        print(f"⚠ AST cache failed: {e}")

_spill_cache = None
def build_spill_cache():
    global _spill_cache
    try:
        _spill_cache =  pd.read_csv('data/spill100_marker_positions_final_9storms.csv')
        print("Loaded marker points")
    except Exception as e:
        print(f"failed to load marker positions")



# starting
build_ast_cache()
build_spill_cache()
