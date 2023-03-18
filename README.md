# model_info_2d

## introduction

Tool to convert atmospheric model (e.g. WRF, NAQPMS) grid IDs to latitude and longitude and vice versa.

There are two main functions:

1. tranfrom model ID to lonlat and back.
2. get XLON, XLAT in wrfout with out wrfout file.

Everything start from a class, and you should create an instance by `model = model_info_2d()`, always you should give some kwags like `proj` (projection), `nx`, `ny`, `dx`, `dy`. Maybe it is cumbersome to build it every time, so, you can see the part of model_info_2d instance to create your personal model_info.

Email: sola1211582551@outlook.com

## function

And this tool provide some useful function like:

1. `ix, iy = model.grid_id(longitude, latitude)`: Gives a lonlat location and returns its nearest integer ID. You can also use function `model.grid_id_float(longitude, latitude)` to get the precise ID in float. Note that ix/iy starts at 0. 

2. `longitude, latitude = model.grid_lonlat(ix, iy)`: similar like `model.grid_id()` but give the ID and return lonlat.

3. `ix_array, iy_array = model.grid_ids(longitude_array, latitude_array)`: similar like `model.grid_id()` but you can give a `list/numpy.ndarray` and return `numpy.ndarray`. If you have lots of points to convert, it will be faster. In fact, if you give a array to `grid_id()`, it will call function `gird_ids()`. It also has a float version `model.grid_ids_float()`.

4. `lons, lats = model.grid_lonlats(ix_array, iy_array)`: similar like `model.grid_lonlat()` but you can give a `list/numpy.ndarray` and return `numpy.ndarray`, like `model.grid_ids()`.

5. `xlon, xlat = model.get_grid()`: return XLONG and XLAT that like the same name variable in wrfout file. It is useful in drawing.

6. `xlon, xlat = model.get_density_grid(density=10)`: return XLONG and XLAT. But there will be a larger matrix. The key ward `density` determine how many point in an original grid. For example, if `density=10`, the `xlon.shape` will be `(10, 10, ny, nx)`, that means every grid has been divided in 10*10 subgrids and their lonlat is `xlon[iy_sub, ix_sub, iy, ix]`, `xlat[iy_sub, ix_sub, iy, ix]`.

## model_info_2d instance

To build your personal model_info, you should have a porjection. The projection can be `cartopy.crs` (ccrs) like `ccrs.PlateCarree()`, or an instance with method like `transform_point`, `grid_id_float`. I provide an example case in `proj_info.py`, it is the projection algorithm from WRF/WPS and I rewrite it in Python (just proj_LC, lambert projection), because I find that there is a little different between `ccrs.LambertConformal()` and wrfout result. 

Notise: i modify the code to let the ID start at 0 (like [1, 1] in Fortran -> [0, 0] in Python).

And then, Use your proj to build a subclass of model_info_2d, like this:

```Python
from model_info_2d import model_info_2d, proj_info

class china_15km(model_info_2d):
    def __init__(self, fun_sim_dat_dir=None):
        truelat1, truelat2 = 40, 25
        stdlon = 105
        lon1, lat1 = 105, 34
        nx, ny = 432, 339
        dx, dy = 15000, 15000
        proj = proj_info.proj_LC(truelat1=truelat1, truelat2=truelat2,
                                 lat1=lat1, lon1=lon1, stdlon=stdlon,
                                 dx=dx, dy=dy, nx=nx, ny=ny)
        lowerleft_lonlat = proj.grid_lonlat(0, 0)
        super().__init__(proj, nx, ny, dx, dy, lowerleft_lonlat)
```

That describe a 15km grid distance domian of East Asia, China. With true latitude of 25° and 40°. The truelat1, truelat2, stdlon(stand_lon), lon1(ref_lon), lat1(ref_lat), dx, dy is same as namelist.wps in WPS. But notise, **the nx and ny is the UNSTAG number (e_we - 1 and e_sn -1 in namelist.wps)**.

You can also define some other function like read_data:

```Python
    ...
    def read_data(self, timestamp, var, layer=None):
        import netCDF4 as nc
        with nc.Dataset(self.fun_sim_dat_dir(timestamp)) as nf:
            if layer is None:
                result = nf[var][0]
            else:
                result = nf[var][0, layer]
        return result
```

Everything is depend on you. Good luck!
