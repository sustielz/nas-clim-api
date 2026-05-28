#!/bin/bash
set -e

mkdir -p data

# Wave Surge data
if [ ! -d "data/return_water_levels.zarr" ]; then
    echo "==> Extracting wave surge data..."
    tar -xzf return_water_levels.tar.gz -C data/
    rm return_water_levels.tar.gz
fi

# Tank surge data
if [ ! -d "data/return_tank_levels.zarr" ]; then
    echo "==> Extracting wave surge data..."
    tar -xzf return_tank_levels.tar.gz -C data/
    rm return_tank_levels.tar.gz
fi

# Fragility data
if [ ! -d "data/fragility_1942.zarr" ]; then
    echo "==> Extracting fragility data..."
    tar -xzf fragility_1942.tar.gz -C data/
    rm fragility_1942.tar.gz
fi

if [ ! -d "data/spill100_marker_positions_final_9storms.csv" ]; then
    cp spill100_marker_positions_final_9storms.csv data/
fi

if [ ! -d "data/100yrfragility.pkl" ]; then
    cp 100yrfragility.pkl data/
fi

# check and start 
echo "==> Starting API..."
ls -la data/

uvicorn main:app --host 0.0.0.0 --port 8080
