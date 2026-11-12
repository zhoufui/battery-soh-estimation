# Data Sources

This repository uses public lithium-ion battery aging datasets. Raw data files are not redistributed here. Processed feature tables used by the experiments are included under `data/processed/`.

## NASA Prognostics Center of Excellence Battery Data Set

Official repository page:

https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Battery data direct link used during this project:

https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip

Cells used in the manuscript:

- B0005
- B0006
- B0007
- B0018

To rebuild the processed NASA feature table, place the raw ZIP file at:

```text
data/raw/NASA_Battery_Data_Set.zip
```

Then run:

```powershell
python src\prepare_nasa_soh.py
```

Output:

```text
data/processed/nasa_soh_features.csv
```

## CALCE Battery Data Archive

Official data archive:

https://web.calce.umd.edu/batteries/data/

The [University of Maryland CS2 dataset description](https://calce.umd.edu/battery-data) lists these contributor publications for the CS2 data:

- He, W.; Williard, N.; Osterman, M.; Pecht, M. Prognostics of lithium-ion batteries based on Dempster-Shafer theory and the Bayesian Monte Carlo method. *Journal of Power Sources* 2011, 196(23), 10314-10321.
- Xing, Y.; Ma, E.; Tsui, K. L.; Pecht, M. An ensemble model for predicting the remaining useful performance of lithium-ion batteries. *Microelectronics Reliability* 2013, 53(6), 811-820.
- Williard, N.; He, W.; Osterman, M.; Pecht, M. Comparative analysis of features for determining state of health in lithium-ion batteries. *International Journal of Prognostics and Health Management* 2013, 4, 1-7.

Cells used in the manuscript:

- CS2_35
- CS2_36

To rebuild the processed CALCE feature table, place the raw CS2_35 and CS2_36 ZIP files under:

```text
data/raw/CALCE/
```

Then run:

```powershell
python src\prepare_calce_soh.py --batteries CS2_35,CS2_36
```

Output:

```text
data/processed/calce_soh_features.csv
```

## Notes on Processed Features

The processed feature tables contain cycle-level health indicators extracted from discharge, charge, and impedance or resistance records. They are intended for reproducibility of the reported machine-learning experiments and should not be interpreted as official releases from NASA or CALCE.
