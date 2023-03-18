# model_info_2d

## introduction

Tool to convert atmospheric model (e.g. WRF, NAQPMS) grid IDs to latitude and longitude and vice versa.

And this tool provide some useful function like:

1. `ix, iy = grid_id(longitude, latitude)`: Gives a lonlat location and returns its nearest integer ID. Note that ix/iy starts at 0.
  You can also use function `grid_id_float(longitude, latitude)` to get the precise ID in float.
  
2. 

2. get_grid(): return XLONG and XLAT that like the same name variable in wrfout file.

3. 
