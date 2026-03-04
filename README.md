# An Xarray Backend using GDAL Bindings

An experimental custom [Xarray Backend](https://docs.xarray.dev/en/stable/internals/how-to-add-new-backend.html) to utilize the [GDAL MultiDimensional API](https://gdal.org/en/stable/user/multidim_raster_data_model.html#multidim-raster-data-model)

## Motivating example

[USDA SOLUS data](https://www.nrcs.usda.gov/resources/data-and-reports/soil-landscapes-of-the-united-states-solus#solus100) - soil properties at various depths in the United States. If we build a GDAL MultiDimensional VRT that combines all the individual COGs into a single virtual dataset, we can read it in with Xarray and the xgdal backend as a single 400GB xarray Dataset with 7 depth layers. The VRT is a lightweight XML 'catalog' that points to the original COGs. Behind the scenes when data is read, GDAL's COG driver is used.


```python
ds_mdim = xr.open_dataset('combined_mdim.vrt', mask_and_scale=True, engine='xgdal')
ds_mdim
```

```
<xarray.Dataset> Size: 400GB
Dimensions:       (y: 31390, x: 49810, depth: 7)
Coordinates:
  * y             (y) float64 251kB 3.258e+06 3.258e+06 ... 1.192e+05 1.19e+05
  * x             (x) float64 398kB -2.54e+06 -2.54e+06 ... 2.441e+06 2.441e+06
  * depth         (depth) int64 56B 0 5 15 30 60 100 150
  * spatial_ref   int64 8B 0
Data variables:
    anylithicdpt  (y, x) float32 6GB ...
    cec7          (depth, y, x) float64 88GB ...
    claytotal     (depth, y, x) float32 44GB ...
    dbovendry     (depth, y, x) float64 88GB ...
    ph1to1h2o     (depth, y, x) float64 88GB ...
    sandtotal     (depth, y, x) float32 44GB ...
    silttotal     (depth, y, x) float32 44GB ...
Indexes:
    spatial_ref  CRSIndex (crs=PROJCS["unknown",GEOGCS["NAD83",DATUM["North_American_Datum_198 ...)
  ┌ x            RasterIndex (crs=PROJCS["unknown",GEOGCS["NAD83",DATUM["North_American_Datum_198 ...)
  └ y
```


## Install

I recommend using [Pixi](https://pixi.prefix.dev/latest/installation/)

```bash
gh repo clone scottyhq/xgdal
pixi install -e .[dev]
# open vscode and run example.ipynb and example-mdim.ipynb
```

## Single File API

Single rasters can still be read specifying `engine='xgdal'`.

```python
import xarray as xr
import xgdal

with xgdal.Env(GS_NO_SIGN_REQUEST=True,
               GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):

    # Read 2D raster w/ GDAL API
    da = xr.open_dataarray(
        'gs://solus100pub/sandtotal_0_cm_p.tif',
        mask_and_scale=True,
        engine='xgdal'
    )
}
```

See more in [example.ipynb](./example.ipynb)


## Multidim API

See also:
https://gdal.org/en/stable/api/python/mdim_api.html
https://gdal.org/en/stable/user/multidim_raster_data_model.html
https://gdal.org/en/stable/drivers/raster/vrt_multidimensional.html

### Constructing multidim VRTs

1. You can use [GDAL mdim CLI](https://gdal.org/en/stable/programs/gdal_mdim.html).

1. This library also includes a method to go from an xarray Dataset to a GDAL MultiDimensional VRT XML string, which can be written out to a .vrt file and read back in with the xgdal backend! With inspiration from [virtualizarr](https://virtualizarr.readthedocs.io/en/stable/index.html) and [xstac](https://github.com/stac-utils/xstac).

See [example-mdim.ipynb](./example-mdim.ipynb) for creating [combined_mdim.vrt](./combined_mdim.vrt)

## Motivation

1. Use [GDAL Python API](https://gdal.org/en/stable/api/python/index.html) directly as an Xarray Backend. Not as a replacement for rioxarray, which uses rasterio and therefore a slightly different API.
    1. For example, we can directly use the [GDAL MultiDimensional API](https://gdal.org/en/stable/user/multidim_raster_data_model.html#multidim-raster-data-model) to read combined VRTs that combine multiple files into a single xarray Dataset.

1. Test out newer Xarray indexing functionality to handle CRS and Affine-backed coordinates such as [xproj](https://xproj.readthedocs.io/en/latest/) and [rasterix](https://github.com/xarray-contrib/rasterix).

1. Seemed like a fun project to get more proficient with Claude Code :)

1. I was working on a project that used [USDA SOLUS data](https://www.nrcs.usda.gov/resources/data-and-reports/soil-landscapes-of-the-united-states-solus#solus100), which conveniently are public COGs on Google Cloud, but I wanted to easily read a full catalog of them rather than loading them one at a time. Options included 1. creating STAC catalog and working with [odc.stac](https://github.com/opendatacube/odc-stac), 2. creating a virtualized Zarr dataset with [virtualizarr](https://virtualizarr.readthedocs.io/en/stable/index.html), or 3. using the GDAL MultiDimensional API to create a combined [multidimensional VRT](https://gdal.org/en/stable/drivers/raster/vrt_multidimensional.html).

### Related projects

[https://github.com/mdsumner/gdx](https://github.com/mdsumner/gdx) - @mdsumner is often at the cutting edge of GDAL, and I was inspired by some of his ideas on [Pangeo Discourse](https://discourse.pangeo.io) and other GitHub threads.

[rioxarray](https://corteva.github.io/rioxarray/stable/) is by far the most mature and well-tested backend for working with rasters in Xarray. Use that! It's possible some of the pieces here (xproj + rasterix) in particular will make there way into rioxarray. Follow https://github.com/corteva/rioxarray/issues/908.


### Roadmap

None! But this is a playground for now and offers some useful stuff, but is mainly for learning and generating ideas.
