"""Xarray accessor for xgdal – provides ``da.xgdal.to_raster()``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
import xproj  # noqa: F401  – ensures .proj accessor is available
from osgeo import gdal, osr

if TYPE_CHECKING:
    import pyproj

# Re-use the dtype mapping (inverted: numpy → GDAL)
_NUMPY_TO_GDAL_DTYPE = {
    np.dtype("uint8"): gdal.GDT_Byte,
    np.dtype("int8"): gdal.GDT_Int8,
    np.dtype("uint16"): gdal.GDT_UInt16,
    np.dtype("int16"): gdal.GDT_Int16,
    np.dtype("uint32"): gdal.GDT_UInt32,
    np.dtype("int32"): gdal.GDT_Int32,
    np.dtype("uint64"): gdal.GDT_UInt64,
    np.dtype("int64"): gdal.GDT_Int64,
    np.dtype("float32"): gdal.GDT_Float32,
    np.dtype("float64"): gdal.GDT_Float64,
    np.dtype("complex64"): gdal.GDT_CFloat32,
    np.dtype("complex128"): gdal.GDT_CFloat64,
}


def _get_geotransform(da: xr.DataArray) -> tuple[float, ...] | None:
    """Try to extract a geotransform from RasterIndex, attrs, or coordinates.

    Priority order:
    1. RasterIndex (from rasterix) – most accurate, tracks through subsetting.
    2. Explicit ``geotransform`` attribute stored by the backend.
    3. Reconstruct from 1D x/y coordinates.
    """
    # 1. Prefer RasterIndex (tracks transforms through isel/sel)
    try:
        from rasterix import RasterIndex

        for idx in da.xindexes.values():
            if isinstance(idx, RasterIndex):
                gt_str = idx.as_geotransform()
                return tuple(float(v) for v in gt_str.split())
    except Exception:
        pass

    # 2. Prefer explicit geotransform stored by the backend
    gt = da.attrs.get("geotransform")
    if gt is not None:
        return tuple(gt)

    # 3. Reconstruct from x / y coordinates
    x = da.coords.get("x")
    y = da.coords.get("y")
    if x is not None and y is not None and len(x) >= 2 and len(y) >= 2:
        dx = float(x.values[1] - x.values[0])
        dy = float(y.values[1] - y.values[0])
        origin_x = float(x.values[0]) - dx / 2
        origin_y = float(y.values[0]) - dy / 2
        return (origin_x, dx, 0.0, origin_y, 0.0, dy)

    return None


def _get_crs(da: xr.DataArray) -> pyproj.CRS | None:
    """Try to extract a CRS from the DataArray via xproj's .proj accessor."""
    try:
        crs = da.proj.crs
        return crs
    except Exception:
        return None


@xr.register_dataarray_accessor("xgdal")
class XGDALAccessor:
    """Xarray DataArray accessor providing GDAL-backed raster I/O.

    Access via ``da.xgdal``.

    Examples
    --------
    >>> da = xr.open_dataarray("file.tif", engine="xgdal")
    >>> da.xgdal.to_raster("output.tif")
    """

    def __init__(self, da: xr.DataArray) -> None:
        self._obj = da

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def encoded_nodata(self) -> float | int | None:
        """Return the original nodata / ``_FillValue``, or ``None``.

        After ``mask_and_scale=True``, the value lives in ``encoding``.
        Without decoding, it stays in ``attrs``.
        """
        val = self._obj.encoding.get("_FillValue")
        if val is None:
            val = self._obj.attrs.get("_FillValue")
        return val

    @property
    def nodata(self) -> float | int | None:
        """Return the effective nodata value.

        If the data has been masked (mask_and_scale), this returns NaN for
        float types.  Otherwise falls back to ``encoded_nodata``.
        """
        enc = self.encoded_nodata
        if enc is not None and np.issubdtype(self._obj.dtype, np.floating):
            # When mask_and_scale was applied, nodata pixels become NaN
            if np.isnan(self._obj.values).any():
                return float("nan")
        return enc

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def to_raster(
        self,
        path: str,
        *,
        driver: str = "GTiff",
        nodata: float | int | None = ...,
        dtype: np.dtype | str | None = None,
    ) -> None:
        """Write this DataArray to a GeoTIFF (or other GDAL raster).

        Parameters
        ----------
        path : str
            Output file path.
        driver : str, default ``"GTiff"``
            GDAL driver short name.
        nodata : float | int | None, optional
            Nodata value to embed in the output file.  By default the
            original ``encoded_nodata`` (``_FillValue``) is used so the
            round-trip preserves the source nodata value.  Pass ``None``
            to explicitly omit a nodata value.
        dtype : np.dtype or str, optional
            Output data type.  Defaults to the *encoded* dtype (i.e. the
            on-disk dtype before mask_and_scale was applied).
        """
        da = self._obj

        # --- Resolve nodata ---
        if nodata is ...:
            nodata = self.encoded_nodata

        # --- Resolve dtype ---
        # Prefer the encoded dtype (the original on-disk type)
        enc_dtype = da.encoding.get("dtype")
        if dtype is not None:
            out_dtype = np.dtype(dtype)
        elif enc_dtype is not None:
            out_dtype = np.dtype(enc_dtype)
        else:
            out_dtype = da.dtype

        gdal_dtype = _NUMPY_TO_GDAL_DTYPE.get(out_dtype, gdal.GDT_Float64)

        # --- Resolve data (unmask NaN → original nodata when possible) ---
        values = da.values.copy()
        if nodata is not None and np.issubdtype(values.dtype, np.floating):
            nan_mask = np.isnan(values)
            if nan_mask.any():
                values[nan_mask] = nodata

        # Cast to the output dtype
        values = values.astype(out_dtype, copy=False)

        # --- Determine shape ---
        # Support (band, y, x) or (y, x)
        if values.ndim == 3:
            n_bands, ny, nx = values.shape
        elif values.ndim == 2:
            ny, nx = values.shape
            n_bands = 1
            values = values[np.newaxis, ...]  # always write as 3-D internally
        else:
            raise ValueError(
                f"Expected 2-D (y, x) or 3-D (band, y, x) data, got {values.ndim}-D"
            )

        # --- Create output dataset ---
        drv = gdal.GetDriverByName(driver)
        if drv is None:
            raise ValueError(f"GDAL driver '{driver}' not found")

        out_ds = drv.Create(path, nx, ny, n_bands, gdal_dtype)
        if out_ds is None:
            raise RuntimeError(f"GDAL could not create '{path}'")

        # --- Geotransform ---
        gt = _get_geotransform(da)
        if gt is not None:
            out_ds.SetGeoTransform(gt)

        # --- CRS ---
        crs = _get_crs(da)
        if crs is not None:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(crs.to_wkt())
            out_ds.SetSpatialRef(srs)

        # --- Write bands ---
        for b in range(n_bands):
            band = out_ds.GetRasterBand(b + 1)
            if nodata is not None:
                band.SetNoDataValue(float(nodata))
            band.WriteArray(values[b])

        out_ds.FlushCache()
        out_ds = None  # close


@xr.register_dataset_accessor("xgdal")
class XGDALDatasetAccessor:
    """Xarray Dataset accessor providing GDAL-backed raster I/O.

    Access via ``ds.xgdal``.

    Examples
    --------
    >>> ds.xgdal.to_mdim_vrt("output.vrt")
    """

    def __init__(self, ds: xr.Dataset) -> None:
        self._obj = ds

    def to_mdim_vrt(self, path: str | None = None, *, validate: bool = False) -> str:
        """Write this Dataset to a GDAL Multidimensional VRT.

        The Dataset must be dask-backed (loaded with ``chunks=``) and
        originally opened via the ``xgdal`` engine so that source
        filenames can be recovered from the dask task graph.

        Parameters
        ----------
        path : str, optional
            Output file path.  If provided the VRT XML is also written
            to disk.

        Returns
        -------
        str
            The VRT XML as a string.
        """
        from xgdal.vrt import dataset_to_mdim_vrt

        xml = dataset_to_mdim_vrt(self._obj, validate=validate)

        if path is not None:
            with open(path, "w") as f:
                f.write(xml)

        return xml
