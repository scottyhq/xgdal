"""Xarray backend engine using GDAL's multidimensional API."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
import rasterix
import xarray as xr
import xproj  # noqa: F401  – registers the .proj accessor
from osgeo import gdal
from xarray.backends import BackendEntrypoint
from xarray.core import indexing

if TYPE_CHECKING:
    from collections.abc import Iterable

# Ensure GDAL raises Python exceptions instead of printing errors
gdal.UseExceptions()

# Map GDAL numeric data types to NumPy dtypes
_GDAL_TO_NUMPY_DTYPE = {
    gdal.GDT_Byte: np.dtype("uint8"),
    gdal.GDT_Int8: np.dtype("int8"),
    gdal.GDT_UInt16: np.dtype("uint16"),
    gdal.GDT_Int16: np.dtype("int16"),
    gdal.GDT_UInt32: np.dtype("uint32"),
    gdal.GDT_Int32: np.dtype("int32"),
    gdal.GDT_UInt64: np.dtype("uint64"),
    gdal.GDT_Int64: np.dtype("int64"),
    gdal.GDT_Float32: np.dtype("float32"),
    gdal.GDT_Float64: np.dtype("float64"),
    gdal.GDT_CFloat32: np.dtype("complex64"),
    gdal.GDT_CFloat64: np.dtype("complex128"),
}


def _gdal_type_to_numpy_dtype(gdal_dt: gdal.ExtendedDataType) -> np.dtype:
    """Convert a GDAL ExtendedDataType to a NumPy dtype."""
    numeric_type = gdal_dt.GetNumericDataType()
    return _GDAL_TO_NUMPY_DTYPE.get(numeric_type, np.dtype("float64"))


# ---------------------------------------------------------------------------
# URI / VSI path normalisation
# ---------------------------------------------------------------------------
# Mapping from cloud-provider URI schemes to GDAL virtual-filesystem prefixes.
_CLOUD_TO_VSI = {
    "http://": "/vsicurl/",
    "https://": "/vsicurl/",
    "gs://": "/vsigs/",
    "s3://": "/vsis3/",
    "az://": "/vsiaz/",
    "abfs://": "/vsiaz/",
}

# Reverse mapping: GDAL VSI prefix → cloud URI scheme recognised by
# xarray's ``is_remote_uri`` (i.e. ``scheme://…``).
# When multiple cloud schemes map to the same VSI prefix (e.g. az:// and
# abfs:// both map to /vsiaz/), we pick the shorter/canonical one.
_VSI_TO_CLOUD = {
    "/vsigs/": "gs://",
    "/vsis3/": "s3://",
    "/vsiaz/": "az://",
    "/vsicurl/": "",  # the remainder is already a full URL.
}


def _to_vsi_path(uri: str) -> str:
    """Convert a cloud URI (``gs://``, ``s3://``, …) to a GDAL ``/vsi…`` path.

    If *uri* is already a ``/vsi…`` path or a local path it is returned
    unchanged.

    For ``http://`` and ``https://`` URLs, the full URL is preserved after
    the ``/vsicurl/`` prefix because GDAL's ``/vsicurl/`` handler expects
    a complete URL (e.g. ``/vsicurl/https://example.com/data.tif``).

    Examples
    --------
    >>> _to_vsi_path("gs://bucket/key.tif")
    '/vsigs/bucket/key.tif'
    >>> _to_vsi_path("https://example.com/data.tif")
    '/vsicurl/https://example.com/data.tif'
    >>> _to_vsi_path("/vsigs/bucket/key.tif")
    '/vsigs/bucket/key.tif'
    """
    for scheme, vsi in _CLOUD_TO_VSI.items():
        if uri.startswith(scheme):
            # /vsicurl/ expects the full URL (including scheme) after the prefix
            if vsi == "/vsicurl/":
                return vsi + uri
            return vsi + uri[len(scheme) :]
    return uri


def _to_cloud_uri(vsi_path: str) -> str:
    """Convert a GDAL ``/vsi…`` path back to a cloud URI (``gs://``, …).

    The returned string always uses a ``scheme://`` form that xarray's
    ``is_remote_uri`` recognises, so ``_get_mtime`` will correctly skip
    the ``os.path.getmtime`` call for remote files.

    If *vsi_path* is already a cloud URI or a local path it is returned
    unchanged.

    Examples
    --------
    >>> _to_cloud_uri("/vsigs/bucket/key.tif")
    'gs://bucket/key.tif'
    >>> _to_cloud_uri("/vsicurl/https://example.com/data.tif")
    'https://example.com/data.tif'
    """
    for vsi, scheme in _VSI_TO_CLOUD.items():
        if vsi_path.startswith(vsi):
            return scheme + vsi_path[len(vsi) :]
    return vsi_path


class GDALBackendArray(xr.backends.BackendArray):
    """Lazy-loading array backed by a GDAL MDArray.

    This wraps a GDAL multidimensional array and provides on-demand
    reading via xarray's explicit indexing protocol.

    Parameters
    ----------
    array_path : str
        The full path of the MDArray within the GDAL dataset (e.g. "/temperature").
    filename_or_obj : str
        The filename or VRT XML string used to open the GDAL dataset.
    open_kwargs : dict
        Keyword arguments for gdal.OpenEx (e.g. open_options).
    """

    def __init__(
        self,
        array_path: str,
        filename_or_obj: str,
        open_kwargs: dict | None = None,
    ) -> None:
        self.array_path = array_path
        self.filename_or_obj = filename_or_obj
        self.open_kwargs = open_kwargs or {}
        self.lock = threading.Lock()

        # Read metadata eagerly (shape, dtype) so xarray can build the Variable
        ds = self._open_dataset()
        rg = ds.GetRootGroup()
        md_array = rg.OpenMDArrayFromFullname(self.array_path)
        if md_array is None:
            raise ValueError(f"Could not open MDArray '{self.array_path}'")
        self.shape = tuple(md_array.GetShape())
        self.dtype = _gdal_type_to_numpy_dtype(md_array.GetDataType())

    def _open_dataset(self):
        """Open the GDAL multidimensional dataset."""
        ds = gdal.OpenEx(
            self.filename_or_obj,
            gdal.OF_MULTIDIM_RASTER | gdal.OF_VERBOSE_ERROR,
            **self.open_kwargs,
        )
        if ds is None:
            raise FileNotFoundError(f"GDAL could not open '{self.filename_or_obj}'")
        return ds

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple) -> np.ndarray:
        """Read a slice from the GDAL MDArray (thread-safe).

        Parameters
        ----------
        key : tuple
            Tuple of integers and/or slices.
        """
        with self.lock:
            ds = self._open_dataset()
            rg = ds.GetRootGroup()
            md_array = rg.OpenMDArrayFromFullname(self.array_path)

            ndim = len(self.shape)

            # Normalise key to a full-length tuple
            if key == ():
                key = tuple(slice(None) for _ in range(ndim))

            array_start_idx = []
            count = []
            array_step = []

            for i, k in enumerate(key):
                dim_size = self.shape[i]
                if isinstance(k, slice):
                    start, stop, step = k.indices(dim_size)
                    if step < 0:
                        raise ValueError("Negative step not supported")
                    length = max(0, (stop - start + step - 1) // step)
                    array_start_idx.append(start)
                    count.append(length)
                    array_step.append(step)
                elif isinstance(k, (int, np.integer)):
                    idx = int(k)
                    if idx < 0:
                        idx += dim_size
                    array_start_idx.append(idx)
                    count.append(1)
                    array_step.append(1)
                else:
                    raise TypeError(f"Unsupported index type: {type(k)}")

            result = md_array.ReadAsArray(
                array_start_idx=array_start_idx,
                count=count,
                array_step=array_step,
            )

            # Squeeze out dimensions that were indexed with an integer
            squeeze_axes = []
            for i, k in enumerate(key):
                if isinstance(k, (int, np.integer)):
                    squeeze_axes.append(i)
            if squeeze_axes:
                result = np.squeeze(result, axis=tuple(squeeze_axes))

            return result


class GDALClassicBackendArray(xr.backends.BackendArray):
    """Lazy-loading array backed by a classic GDAL raster dataset.

    Used for single GeoTIFFs and other formats that don't support the
    GDAL multidimensional API. Each band is treated as a separate slice
    along the first dimension (if multi-band) or the array is 2-D (y, x).

    Parameters
    ----------
    filename : str
        Path to the raster file.
    shape : tuple
        Shape of the array, e.g. ``(n_bands, ny, nx)`` or ``(ny, nx)``.
    dtype : np.dtype
        NumPy dtype of the data.
    overview_level : int or None
        If set, read from this overview level instead of the full-resolution
        data. Level 0 is the first (largest) overview, higher levels are
        progressively smaller.
    """

    def __init__(
        self,
        filename: str,
        shape: tuple,
        dtype: np.dtype,
        overview_level: int | None = None,
    ) -> None:
        self.filename = filename
        self.shape = shape
        self.dtype = dtype
        self.overview_level = overview_level
        self.lock = threading.Lock()

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _get_band(self, ds, band_index: int):
        """Return the band (or overview band) for the given 1-based index.

        Parameters
        ----------
        ds : gdal.Dataset
            An open GDAL raster dataset.
        band_index : int
            1-based band index.

        Returns
        -------
        gdal.Band
        """
        band = ds.GetRasterBand(band_index)
        if self.overview_level is not None:
            overview = band.GetOverview(self.overview_level)
            if overview is None:
                n_ovr = band.GetOverviewCount()
                raise ValueError(
                    f"Overview level {self.overview_level} not available. "
                    f"Band {band_index} has {n_ovr} overview(s) "
                    f"(valid levels: 0–{n_ovr - 1})."
                )
            return overview
        return band

    def _raw_indexing_method(self, key: tuple) -> np.ndarray:
        ndim = len(self.shape)

        if key == ():
            key = tuple(slice(None) for _ in range(ndim))

        with self.lock:
            ds = gdal.OpenEx(self.filename, gdal.OF_RASTER | gdal.OF_VERBOSE_ERROR)
            if ds is None:
                raise FileNotFoundError(f"GDAL could not open '{self.filename}'")

            if ndim == 3:
                # (band, y, x)
                band_key, y_key, x_key = key
            else:
                # (y, x)
                band_key = 0  # single band
                y_key, x_key = key

            # Resolve x slice
            nx = self.shape[-1]
            if isinstance(x_key, slice):
                x_start, x_stop, x_step = x_key.indices(nx)
                x_size = max(0, (x_stop - x_start + x_step - 1) // x_step)
            elif isinstance(x_key, (int, np.integer)):
                x_start = int(x_key)
                if x_start < 0:
                    x_start += nx
                x_size = 1
                x_step = 1
            else:
                raise TypeError(f"Unsupported x index type: {type(x_key)}")

            # Resolve y slice
            ny = self.shape[-2]
            if isinstance(y_key, slice):
                y_start, y_stop, y_step = y_key.indices(ny)
                y_size = max(0, (y_stop - y_start + y_step - 1) // y_step)
            elif isinstance(y_key, (int, np.integer)):
                y_start = int(y_key)
                if y_start < 0:
                    y_start += ny
                y_size = 1
                y_step = 1
            else:
                raise TypeError(f"Unsupported y index type: {type(y_key)}")

            # Read band data
            if ndim == 3:
                n_bands = self.shape[0]
                if isinstance(band_key, slice):
                    band_start, band_stop, band_step = band_key.indices(n_bands)
                    band_indices = list(range(band_start, band_stop, band_step))
                elif isinstance(band_key, (int, np.integer)):
                    band_indices = [int(band_key)]
                else:
                    raise TypeError(f"Unsupported band index type: {type(band_key)}")

                arrays = []
                for bi in band_indices:
                    band = self._get_band(ds, bi + 1)  # GDAL bands are 1-indexed
                    arr = band.ReadAsArray(
                        x_start, y_start, x_size * x_step, y_size * y_step
                    )
                    if x_step > 1 or y_step > 1:
                        arr = arr[::y_step, ::x_step]
                    arrays.append(arr)
                result = np.stack(arrays, axis=0)

                # Squeeze integer-indexed dims
                if isinstance(band_key, (int, np.integer)):
                    result = result[0]
                if isinstance(y_key, (int, np.integer)):
                    result = np.squeeze(result, axis=-2 if result.ndim > 1 else 0)
                if isinstance(x_key, (int, np.integer)):
                    result = np.squeeze(result, axis=-1)
            else:
                band = self._get_band(ds, 1)
                result = band.ReadAsArray(
                    x_start, y_start, x_size * x_step, y_size * y_step
                )
                if x_step > 1 or y_step > 1:
                    result = result[::y_step, ::x_step]
                if isinstance(y_key, (int, np.integer)):
                    result = result[0]
                if isinstance(x_key, (int, np.integer)):
                    result = result[..., 0]

            ds = None  # close
            return result


def _open_single_tif(
    filename: str,
    overview_level: int | None = None,
) -> xr.Dataset:
    """Open a single GeoTIFF (or any classic 2D GDAL raster) as an xr.Dataset.

    Uses the classic GDAL raster API with lazy-loaded backend arrays.
    Bands become a ``band`` dimension when there are multiple bands.
    Pixel-centre x/y coordinates are derived from the geotransform.

    Parameters
    ----------
    filename : str
        Path to the raster file.
    overview_level : int or None
        If set, open this overview level instead of the full-resolution data.
        Level 0 is the first (largest) overview.
    """
    classic_ds = gdal.OpenEx(filename, gdal.OF_RASTER | gdal.OF_VERBOSE_ERROR)
    if classic_ds is None:
        raise FileNotFoundError(f"GDAL could not open '{filename}'")

    gt = classic_ds.GetGeoTransform()
    crs_wkt = (
        classic_ds.GetSpatialRef().ExportToWkt() if classic_ds.GetSpatialRef() else ""
    )
    n_bands = classic_ds.RasterCount

    # Get band (or overview) to determine dimensions and dtype
    band = classic_ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    if overview_level is not None:
        n_ovr = band.GetOverviewCount()
        if overview_level < 0 or overview_level >= n_ovr:
            raise ValueError(
                f"overview_level={overview_level} is out of range. "
                f"This file has {n_ovr} overview(s) (valid levels: 0–{n_ovr - 1})."
            )
        overview = band.GetOverview(overview_level)
        nx = overview.XSize
        ny = overview.YSize
        gdal_dtype = overview.DataType
    else:
        nx = classic_ds.RasterXSize
        ny = classic_ds.RasterYSize
        gdal_dtype = band.DataType

    np_dtype = _GDAL_TO_NUMPY_DTYPE.get(gdal_dtype, np.dtype("float64"))

    # Get on-disk block (tile) size for preferred_chunks encoding.
    # GetBlockSize() returns [block_x, block_y].
    if overview_level is not None:
        block_x, block_y = overview.GetBlockSize()
    else:
        block_x, block_y = band.GetBlockSize()

    # Recompute geotransform for the overview resolution.
    # The overview has (nx, ny) pixels covering the same spatial extent as
    # the full-resolution raster, so we scale pixel_width and pixel_height.
    full_nx = classic_ds.RasterXSize
    full_ny = classic_ds.RasterYSize
    ovr_gt = (
        gt[0],  # origin_x stays the same
        gt[1] * full_nx / nx,  # pixel_width scaled
        gt[2],
        gt[3],  # origin_y stays the same
        gt[4],
        gt[5] * full_ny / ny,  # pixel_height scaled
    )

    classic_ds = None  # close

    # Pixel-centre coordinates from (possibly scaled) geotransform
    x_coords = np.arange(nx) * ovr_gt[1] + ovr_gt[0] + ovr_gt[1] / 2
    y_coords = np.arange(ny) * ovr_gt[5] + ovr_gt[3] + ovr_gt[5] / 2

    # Always include a "band" dimension (even for single-band files) to
    # match the convention used by rioxarray / the rasterio backend.
    shape = (n_bands, ny, nx)
    dims = ("band", "y", "x")

    backend_array = GDALClassicBackendArray(filename, shape, np_dtype, overview_level)
    data = indexing.LazilyIndexedArray(backend_array)

    attrs: dict = {}
    encoding: dict = {"dtype": np_dtype}

    # Expose on-disk block sizes as preferred_chunks so that
    # xr.open_dataarray(..., chunks={}) uses tile-aligned chunks.
    preferred_chunks: dict[str, int] = {}
    if block_y > 0:
        preferred_chunks["y"] = block_y
    if block_x > 0:
        preferred_chunks["x"] = block_x
    if n_bands > 1:
        preferred_chunks["band"] = n_bands
    if preferred_chunks:
        encoding["preferred_chunks"] = preferred_chunks

    if nodata is not None:
        attrs["_FillValue"] = nodata

    var = xr.Variable(dims, data, attrs=attrs, encoding=encoding)

    # Match rioxarray (basically da.name = None)
    data_vars = {"__xarray_dataarray_variable__": var}
    coords: dict = {
        "x": xr.Variable("x", x_coords, attrs={"axis": "X"}),
        "y": xr.Variable("y", y_coords, attrs={"axis": "Y"}),
        "band": xr.Variable("band", np.arange(1, n_bands + 1)),
    }

    ds_attrs: dict = {}
    if ovr_gt:
        ds_attrs["geotransform"] = ovr_gt
    if crs_wkt:
        ds_attrs["crs_wkt"] = crs_wkt

    ds = xr.Dataset(data_vars, coords=coords, attrs=ds_attrs)

    return ds


def _open_multidim(filename_or_obj: str) -> xr.Dataset:
    """Open a multidimensional VRT (or any GDAL multidim source) as xr.Dataset.

    This reads all arrays from the root group, discovers dimensions and
    coordinate variables, and builds an xarray Dataset with lazy-loaded data.
    """
    md_ds = gdal.OpenEx(
        filename_or_obj,
        gdal.OF_MULTIDIM_RASTER | gdal.OF_VERBOSE_ERROR,
    )
    if md_ds is None:
        raise FileNotFoundError(
            f"GDAL could not open '{filename_or_obj}' as multidimensional"
        )

    rg = md_ds.GetRootGroup()

    # Discover all dimensions
    gdal_dims = rg.GetDimensions() or []
    dim_info = {}
    for d in gdal_dims:
        dim_info[d.GetName()] = {
            "size": d.GetSize(),
            "type": d.GetType(),
            "indexing_var": d.GetIndexingVariable(),
        }

    array_names = rg.GetMDArrayNames() or []

    # First pass: identify coordinate variables (arrays whose name matches a dimension)
    coord_names = set()
    for name in array_names:
        if name in dim_info:
            coord_names.add(name)

    # Also check indexing variables from dimensions
    for dname, dinfo in dim_info.items():
        iv = dinfo["indexing_var"]
        if iv is not None:
            coord_names.add(iv.GetName())

    data_vars = {}
    coords = {}

    for name in array_names:
        md_array = rg.OpenMDArray(name)
        if md_array is None:
            continue

        dims = md_array.GetDimensions()
        dim_names = tuple(d.GetName() for d in dims)

        backend_array = GDALBackendArray(
            array_path=md_array.GetFullName(),
            filename_or_obj=filename_or_obj,
        )
        data = indexing.LazilyIndexedArray(backend_array)

        attrs = {}
        encoding = {}

        # Expose on-disk block sizes as preferred_chunks.
        # MDArray.GetBlockSize() returns a list of block sizes per dimension.
        block_sizes = md_array.GetBlockSize()
        if block_sizes:
            preferred_chunks = {}
            for dname, bsz in zip(dim_names, block_sizes):
                # A block size of 0 means "entire dimension" — use the
                # dimension size itself so the preferred chunk covers it fully.
                if bsz > 0:
                    preferred_chunks[dname] = bsz
                else:
                    dim_size = md_array.GetDimensions()[
                        dim_names.index(dname)
                    ].GetSize()
                    preferred_chunks[dname] = dim_size
            if preferred_chunks:
                encoding["preferred_chunks"] = preferred_chunks

        nodata = md_array.GetNoDataValue()
        if nodata is not None:
            attrs["_FillValue"] = nodata

        unit = md_array.GetUnit()
        if unit:
            attrs["units"] = unit

        scale = md_array.GetScale()
        offset = md_array.GetOffset()
        if scale is not None and scale != 1.0:
            attrs["scale_factor"] = scale
        if offset is not None and offset != 0.0:
            attrs["add_offset"] = offset

        # Read per-array attributes (e.g. from multidim VRT <Attribute> elements).
        # These supplement the values already obtained via the dedicated GDAL
        # accessors (GetNoDataValue, GetUnit, GetScale, GetOffset).
        for gdal_attr in md_array.GetAttributes() or []:
            attr_name = gdal_attr.GetName()
            if attr_name not in attrs:
                attrs[attr_name] = gdal_attr.Read()

        srs = md_array.GetSpatialRef()
        crs_wkt_str = srs.ExportToWkt() if srs else ""

        var = xr.Variable(dim_names, data, attrs=attrs, encoding=encoding)

        if name in coord_names:
            coords[name] = var
        else:
            data_vars[name] = var

    # Read group-level attributes
    ds_attrs = {}
    group_attrs = rg.GetAttributes() or []
    for attr in group_attrs:
        ds_attrs[attr.GetName()] = attr.Read()

    ds = xr.Dataset(data_vars, coords=coords, attrs=ds_attrs)

    # Stash CRS WKT in attrs for open_dataset to pick up after decode_cf
    if crs_wkt_str:
        ds.attrs["crs_wkt"] = crs_wkt_str

    return ds


def _is_vrt_string(filename_or_obj: str) -> bool:
    """Check if the input looks like an in-memory VRT XML string."""
    return isinstance(filename_or_obj, str) and filename_or_obj.strip().startswith(
        "<VRTDataset"
    )


class XGDALBackendEntrypoint(BackendEntrypoint):
    """Xarray backend entrypoint using GDAL.

    Supports opening:
    - Single GeoTIFF files (classic 2D raster, exposed via GDAL's multidim API)
    - Multidimensional VRT files or in-memory VRT XML strings

    Register as engine ``"xgdal"`` via the ``xarray.backends`` entry point.
    """

    description = "Open raster data with GDAL"
    url = "https://github.com/xarray-contrib/xgdal"

    def open_dataset(
        self,
        filename_or_obj: str,
        *,
        drop_variables: Iterable[str] | None = None,
        mask_and_scale: bool | None = None,
        overview_level: int | None = None,
    ) -> xr.Dataset:
        """Open a raster file as an xarray Dataset using GDAL.

        Parameters
        ----------
        filename_or_obj : str
            Path to a GeoTIFF, VRT file, or an in-memory VRT XML string.
        drop_variables : iterable of str, optional
            Variable names to exclude from the dataset.
        mask_and_scale : bool, optional
            If True, apply CF mask and scale decoding.
        overview_level : int, optional
            If set, read from this overview level instead of the
            full-resolution data. Level 0 is the first (largest) overview,
            higher levels are progressively smaller. Only applies to classic
            raster files (e.g. GeoTIFF), not multidimensional VRTs.
        """
        # Normalise cloud URIs (gs://, s3://, …) → GDAL /vsi paths so that
        # GDAL can open them.  The *original* filename_or_obj (which may be a
        # cloud URI such as gs://…) is what xarray passes to _get_mtime();
        # because it matches ``is_remote_uri``, the mtime lookup is skipped
        # and dask chunking works correctly.
        gdal_path = _to_vsi_path(filename_or_obj)

        # Determine whether input is a multidimensional VRT or a single raster
        if _is_vrt_string(gdal_path):
            ds = _open_multidim(gdal_path)
        elif isinstance(gdal_path, str) and gdal_path.endswith(".vrt"):
            # Could be classic or multidim VRT – try multidim first
            try:
                ds = _open_multidim(gdal_path)
            except Exception:
                ds = _open_single_tif(gdal_path, overview_level=overview_level)
        else:
            ds = _open_single_tif(gdal_path, overview_level=overview_level)

        # Drop requested variables
        if drop_variables is not None:
            ds = ds.drop_vars([v for v in drop_variables if v in ds])

        # Apply CF mask & scale decoding when requested.
        # _FillValue is stored in variable attrs by the open helpers above;
        # decode_cf moves it to encoding and replaces matching values with NaN.
        if mask_and_scale is not None and mask_and_scale:
            ds = xr.decode_cf(ds, mask_and_scale=True)

        # Assign CRS via xproj *after* decode_cf (which would drop the
        # CRSIndex).  The helpers stash the WKT in ds.attrs["crs_wkt"].
        crs_wkt = ds.attrs.pop("crs_wkt", None)
        if crs_wkt:
            ds = ds.proj.assign_crs(spatial_ref=crs_wkt)

        # Store the GeoTransform on the spatial_ref coordinate so that
        # rasterix.assign_index can discover it (priority 1 heuristic).
        gt = ds.attrs.get("geotransform")
        if gt is not None and "spatial_ref" in ds.coords:
            gt_str = " ".join(str(v) for v in gt)
            ds["spatial_ref"].attrs["GeoTransform"] = gt_str

        # Assign a RasterIndex via rasterix for affine-transform-aware
        # coordinate indexing.  assign_index auto-detects the transform
        # from the GeoTransform attribute on spatial_ref (preferred) or
        # falls back to the 1D x/y coordinate arrays.
        # Detect spatial dimension names (lowercase or uppercase).
        x_dim = y_dim = None
        for candidate_x, candidate_y in [("x", "y"), ("X", "Y")]:
            if candidate_x in ds.dims and candidate_y in ds.dims:
                x_dim, y_dim = candidate_x, candidate_y
                break
        if x_dim is not None:
            ds = rasterix.assign_index(ds, x_dim=x_dim, y_dim=y_dim)

        ds.set_close(lambda: None)
        return ds

    def guess_can_open(self, filename_or_obj: str) -> bool:
        """Return True if this backend can likely open the given input."""
        if _is_vrt_string(filename_or_obj):
            return True
        # Normalise cloud URIs so the extension check works for paths
        # like gs://bucket/file.tif or /vsigs/bucket/file.tif
        path = _to_vsi_path(str(filename_or_obj))
        try:
            _, ext = path.rsplit(".", 1)
        except (ValueError, AttributeError):
            return False
        return ext.lower() in {"tif", "tiff", "vrt"}
