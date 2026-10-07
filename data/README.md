# Data

Two datasets are used. Only the weather file is stored in this repository.

## Weather: included
`raw/POWER_Point_Hourly_20200622_20230630_047d27N_002d36E_LST.csv`
Hourly meteorological data from the [NASA POWER](https://power.larc.nasa.gov/) project for a single central point in
France (latitude 47.2666, longitude 2.3615), 22 June 2020 to 30 June 2023. NASA POWER data is freely available;
please cite NASA POWER when you reuse it. The point is an assumption: the production data does not say where the
generating sites are.

## Production: download it yourself
`raw/intermittent-renewables-production-france.csv`
Hourly wind and solar production for the French grid (2020 to mid 2023), from the Kaggle dataset
[Wind and Solar Daily Power Production](https://www.kaggle.com/datasets/henriupton/wind-solar-electricity-production).
It is not redistributed here because its licence is set by the original publisher. Download it, save it at the path
above (or pass `--production` to the scripts), and re-run the pipeline.

The file has these columns: `Date and Hour` (timezone-aware, start of the hour), `Date`, `StartHour`, `EndHour`,
`Source` (Wind or Solar), `Production` (MWh), `dayOfYear`, `dayName`, `monthName`.
