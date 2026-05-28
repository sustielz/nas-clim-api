FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install gdown

COPY main.py data_sets.py ./
COPY return_water_levels.tar.gz ./
COPY return_tank_levels.tar.gz ./
COPY fragility_1942.tar.gz ./
COPY spill100_marker_positions_final_9storms.csv ./
COPY 100yrfragility.pkl ./

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
