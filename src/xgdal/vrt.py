"""Build a multidimensional VRT XML string from individual raster files."""

from __future__ import annotations

import functools
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
import xmlschema
import xarray as xr

_GDAL_VRT_XSD_URL = "https://raw.githubusercontent.com/OSGeo/gdal/master/frmts/vrt/data/gdalvrt.xsd"


@functools.lru_cache(maxsize=1)
def _gdal_vrt_schema() -> xmlschema.XMLSchema:
    return xmlschema.XMLSchema(_GDAL_VRT_XSD_URL)


# Map NumPy dtypes to GDAL type name strings used in VRT XML
_NUMPY_DTYPE_TO_GDAL_NAME: dict[np.dtype, str] = {
    np.dtype("uint8"): "Byte",
    np.dtype("int8"): "Int8",
    np.dtype("uint16"): "UInt16",
    np.dtype("int16"): "Int16",
    np.dtype("uint32"): "UInt32",
    np.dtype("int32"): "Int32",
    np.dtype("uint64"): "UInt64",
    np.dtype("int64"): "Int64",
    np.dtype("float32"): "Float32",
    np.dtype("float64"): "Float64",
    np.dtype("complex64"): "CFloat32",
    np.dtype("complex128"): "CFloat64",
}



# ---------------------------------------------------------------------------
# Helpers for dataset_to_mdim_vrt
# ---------------------------------------------------------------------------

_SPATIAL_DIM_NAMES = {"x", "y", "X", "Y"}


def _prettify_xml(root: Element) -> str:
    """Serialise an ElementTree to a pretty-printed XML string (no declaration)."""
    rough = tostring(root, encoding="unicode")
    pretty = parseString(rough).toprettyxml(indent="  ")
    lines = pretty.split("\n")
    if lines[0].startswith("<?xml"):
        lines = lines[1:]
    return "\n".join(lines)


def _gdal_type_name(dtype: np.dtype) -> str:
    """Return the GDAL type name string for a NumPy dtype."""
    return _NUMPY_DTYPE_TO_GDAL_NAME.get(np.dtype(dtype), "Float32")


def _coord_gdal_type(values: np.ndarray) -> str:
    """Return a GDAL type name suitable for a coordinate array."""
    if np.issubdtype(values.dtype, np.integer):
        return "Int64"
    elif np.issubdtype(values.dtype, np.floating):
        return "Float64"
    return "String"


def _is_regularly_spaced(values: np.ndarray, rtol: float = 1e-6) -> bool:
    """Return True if *values* are evenly spaced."""
    if len(values) < 2:
        return True
    diffs = np.diff(values.astype(float))
    return bool(np.allclose(diffs, diffs[0], rtol=rtol))


def _gdal_attribute_type(value: object) -> str:
    """Return a GDAL DataType string suitable for an ``<Attribute>`` element."""
    if isinstance(value, (int, np.integer)):
        # Pick the smallest signed integer type that fits
        if -(2**15) <= value <= 2**15 - 1:
            return "Int16"
        if -(2**31) <= value <= 2**31 - 1:
            return "Int32"
        return "Int64"
    if isinstance(value, (float, np.floating)):
        return "Float64"
    return "String"


# Attributes that xarray may move to encoding after decoding; check both
# da.attrs and da.encoding for these.
_ENCODING_FALLBACK: set[str] = {"scale_factor", "add_offset"}

# Attrs emitted as dedicated GDAL elements (Unit/Offset/Scale) at specific
# positions in the Array element; skip them in the generic Attribute loop.
_GDAL_INLINE_ATTRS: set[str] = {"scale_factor", "add_offset", "units"}


def _build_array_attributes(
    parent: Element,
    da: xr.DataArray,
) -> None:
    """Append ``<Attribute>`` sub-elements to *parent* for all DataArray attrs.

    Emits every entry in ``da.attrs``.  For ``scale_factor``, ``add_offset``,
    and ``_FillValue``, also checks ``da.encoding`` as a fallback (xarray
    moves these there after mask-and-scale decoding).
    """
    seen: set[str] = set()
    names_values: list[tuple[str, object]] = []

    for attr_name, value in da.attrs.items():
        if attr_name == "_FillValue":
            continue  # emitted as <NoDataValue> by the caller
        seen.add(attr_name)
        names_values.append((attr_name, value))

    # Encoding fallback for CF-encoding attrs not present in da.attrs
    for attr_name in _ENCODING_FALLBACK:
        if attr_name not in seen:
            value = da.encoding.get(attr_name)
            if value is not None:
                names_values.append((attr_name, value))

    for attr_name, value in names_values:
        if attr_name in _GDAL_INLINE_ATTRS:
            continue  # emitted inline at the correct position by the caller
        attr_el = SubElement(parent, "Attribute", name=attr_name)
        SubElement(attr_el, "DataType").text = _gdal_attribute_type(value)
        SubElement(attr_el, "Value").text = str(value)


def _extract_source_files(da: xr.DataArray, stack_dim: str) -> list[str]:
    """Extract per-slice source filenames from a dask-backed DataArray.

    Walks the dask graph to find ``GDALClassicBackendArray`` objects and
    returns a list of filenames ordered along *stack_dim*.

    Raises ``ValueError`` if the DataArray is not dask-backed or the
    backend arrays cannot be found.
    """
    import dask.array as dask_array

    from xgdal.backend import GDALClassicBackendArray

    if not isinstance(da.data, dask_array.Array):
        raise ValueError(
            f"Variable '{da.name}' is not dask-backed; "
            "cannot auto-detect source files from the task graph. "
            "Load data with chunks= to enable dask."
        )

    graph = dict(da.data.__dask_graph__())

    # Collect concatenation entries: (stacking_index, referenced_key_prefix)
    concat_entries: list[tuple[int, str]] = []
    for key in graph:
        if isinstance(key, tuple) and "concatenate" in str(key[0]):
            stack_idx = key[1]  # index along the concatenation dimension
            # The value is a tuple whose first element is the referenced key prefix
            ref = graph[key]
            if isinstance(ref, tuple) and len(ref) >= 1:
                ref_prefix = ref[0] if isinstance(ref[0], str) else str(ref[0])
            else:
                continue
            concat_entries.append((stack_idx, ref_prefix))

    # If there are no concatenate keys, we may have a single-slice array.
    # Look directly for original- keys.
    if not concat_entries:
        filenames: list[str] = []
        for key, val in graph.items():
            if isinstance(key, str) and key.startswith("original-"):
                backend = _unwrap_backend_array(val)
                if isinstance(backend, GDALClassicBackendArray):
                    filenames.append(backend.filename)
        if filenames:
            return filenames
        raise ValueError(
            f"Could not find source files in the dask graph for variable '{da.name}'."
        )

    # Sort by stacking index
    concat_entries.sort(key=lambda x: x[0])

    # For each concat entry, find the original- key via the open_dataset dependency chain
    # Only keep the first filename for each stacking index (logical slice)
    index_to_filename = {}
    for stack_idx, ref_prefix in concat_entries:
        if stack_idx in index_to_filename:
            continue  # already have a file for this logical slice
        original_key = f"original-{ref_prefix}"
        if original_key not in graph:
            raise ValueError(
                f"Could not find original key '{original_key}' in dask graph."
            )
        val = graph[original_key]
        backend = _unwrap_backend_array(val)
        if not isinstance(backend, GDALClassicBackendArray):
            raise ValueError(
                f"Expected GDALClassicBackendArray, got {type(backend)} "
                f"for key '{original_key}'."
            )
        fname = backend.filename
        index_to_filename[stack_idx] = fname

    # Return filenames ordered by stacking index
    return [index_to_filename[i] for i in sorted(index_to_filename)]


def _unwrap_backend_array(val: object) -> object:
    """Unwrap xarray indexing adapters to get the innermost backend array.

    The chain is typically:
    ImplicitToExplicitIndexingAdapter → CopyOnWriteArray → LazilyIndexedArray → BackendArray
    """
    # Walk .array attributes until we can't go deeper
    current = val
    for _ in range(10):  # safety limit
        if hasattr(current, "array"):
            current = current.array
        else:
            break
    return current


# ---------------------------------------------------------------------------
# Main: Dataset → Multidimensional VRT
# ---------------------------------------------------------------------------

# NOTE: this builds the XML from scratch by going over the Xarray Dataset
# an alternative would be to construct an in-memory GDAL MDIM dataset and then
# use GDAL to write to VRT/XML
# see: https://github.com/OSGeo/gdal/blob/836f84be6c247e47dd879ed53cae8f0f17dca81d/autotest/gdrivers/vrtmultidim.py#L1228
# This is easy for now
def dataset_to_mdim_vrt(ds: xr.Dataset, validate: bool = False) -> str:
    """Build a multidimensional VRT XML string from an xarray Dataset.

    The Dataset should contain:
    - Spatial coordinates named ``x``/``X`` and ``y``/``Y``.
    - One or more non-spatial dimensions (e.g. ``depth``, ``time``).
    - Dask-backed data variables originally loaded via the ``xgdal`` engine,
      so that source filenames can be recovered from the dask task graph.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset.

    Returns
    -------
    str
        The VRT XML as a string.
    """
    # ── Identify spatial vs non-spatial dimensions ─────────────────
    x_dim = y_dim = None
    for cx, cy in [("x", "y"), ("X", "Y")]:
        if cx in ds.dims and cy in ds.dims:
            x_dim, y_dim = cx, cy
            break
    if x_dim is None:
        raise ValueError(
            "Dataset must have spatial dimensions named 'x'/'y' or 'X'/'Y'."
        )

    x_coords = ds.coords[x_dim].values
    y_coords = ds.coords[y_dim].values
    nx = len(x_coords)
    ny = len(y_coords)

    # Non-spatial dimensions (everything except x, y, and any scalar coords)
    non_spatial_dims = [
        d for d in ds.dims if d not in {x_dim, y_dim} and d not in _SPATIAL_DIM_NAMES
    ]

    # ── CRS ────────────────────────────────────────────────────────
    crs_wkt = ""
    try:
        crs_wkt = str(ds.proj.crs)
    except Exception:
        pass

    # ── Build XML ──────────────────────────────────────────────────
    vrt_root = Element("VRTDataset")
    group = SubElement(vrt_root, "Group", name="/")

    # Emit <Dimension> elements
    for dim_name in non_spatial_dims:
        dim_attrs = {"name": dim_name, "size": str(ds.sizes[dim_name])}
        if dim_name in ds.coords:
            dim_attrs["indexingVariable"] = dim_name
        SubElement(group, "Dimension", **dim_attrs)

    SubElement(
        group, "Dimension", name=y_dim, size=str(ny),
        type="HORIZONTAL_Y", direction="NORTH", indexingVariable=y_dim,
    )
    SubElement(
        group, "Dimension", name=x_dim, size=str(nx),
        type="HORIZONTAL_X", direction="EAST", indexingVariable=x_dim,
    )

    # ── Coordinate arrays ──────────────────────────────────────────

    # Non-spatial coordinate arrays (e.g. depth, time)
    for dim_name in non_spatial_dims:
        if dim_name in ds.coords:
            vals = ds.coords[dim_name].values
            coord_type = _coord_gdal_type(vals)
            arr_el = SubElement(group, "Array", name=dim_name)
            SubElement(arr_el, "DataType").text = coord_type
            SubElement(arr_el, "DimensionRef", ref=dim_name)
            SubElement(arr_el, "InlineValues").text = " ".join(str(v) for v in vals)

    # Y coordinate
    y_arr = SubElement(group, "Array", name=y_dim)
    SubElement(y_arr, "DataType").text = "Float64"
    SubElement(y_arr, "DimensionRef", ref=y_dim)
    if _is_regularly_spaced(y_coords):
        dy = float(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0
        SubElement(y_arr, "RegularlySpacedValues", start=str(y_coords[0]), increment=str(dy))
    else:
        SubElement(y_arr, "InlineValues").text = " ".join(str(v) for v in y_coords)

    # X coordinate
    x_arr = SubElement(group, "Array", name=x_dim)
    SubElement(x_arr, "DataType").text = "Float64"
    SubElement(x_arr, "DimensionRef", ref=x_dim)
    if _is_regularly_spaced(x_coords):
        dx = float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0
        SubElement(x_arr, "RegularlySpacedValues", start=str(x_coords[0]), increment=str(dx))
    else:
        SubElement(x_arr, "InlineValues").text = " ".join(str(v) for v in x_coords)

    # ── Data arrays (one per data variable) ────────────────────────
    for var_name in ds.data_vars:
        da = ds[var_name]

        # Skip scalar or non-spatial variables (e.g. spatial_ref)
        if x_dim not in da.dims or y_dim not in da.dims:
            continue

        # Determine the dimension order: non-spatial dims first, then y, x
        var_non_spatial = [d for d in da.dims if d not in {x_dim, y_dim}]

        # Determine encoded dtype (prefer encoding over runtime dtype).
        # Default to Float32 when no encoded dtype is available.
        enc_dtype = da.encoding.get("dtype")
        if enc_dtype is not None:
            dtype_name = _gdal_type_name(enc_dtype)
        else:
            dtype_name = _gdal_type_name(da.dtype)

        data_arr = SubElement(group, "Array", name=var_name)
        SubElement(data_arr, "DataType").text = dtype_name

        for d in var_non_spatial:
            SubElement(data_arr, "DimensionRef", ref=d)
        SubElement(data_arr, "DimensionRef", ref=y_dim)
        SubElement(data_arr, "DimensionRef", ref=x_dim)

        preferred_chunks = da.encoding.get("preferred_chunks")
        if preferred_chunks:
            all_dims = var_non_spatial + [y_dim, x_dim]
            sizes = [str(preferred_chunks.get(d, 1)) for d in all_dims]
            SubElement(data_arr, "BlockSize").text = ",".join(sizes)

        if crs_wkt:
            srs_el = SubElement(data_arr, "SRS", dataAxisToSRSAxisMapping="2,1")
            srs_el.text = crs_wkt

        unit = da.attrs.get("units")
        if unit is None:
            unit = da.encoding.get("units")
        if unit is not None:
            SubElement(data_arr, "Unit").text = str(unit)

        nodata = da.attrs.get("_FillValue")
        if nodata is None:
            nodata = da.encoding.get("_FillValue")
        if nodata is not None:
            SubElement(data_arr, "NoDataValue").text = str(nodata)

        offset = da.attrs.get("add_offset")
        if offset is None:
            offset = da.encoding.get("add_offset")
        if offset is not None:
            SubElement(data_arr, "Offset").text = str(offset)

        scale = da.attrs.get("scale_factor")
        if scale is None:
            scale = da.encoding.get("scale_factor")
        if scale is not None:
            SubElement(data_arr, "Scale").text = str(scale)

        # Extract source files from the dask graph
        # This assumes a single stacking dimension (the first non-spatial dim)
        if var_non_spatial:
            stack_dim = var_non_spatial[0]
            source_files = _extract_source_files(da, stack_dim)
            n_stack = ds.sizes[stack_dim]

            if len(source_files) != n_stack:
                raise ValueError(
                    f"Found {len(source_files)} source files for variable "
                    f"'{var_name}' but dimension '{stack_dim}' has size {n_stack}."
                )

            for i, f in enumerate(source_files):
                source = SubElement(data_arr, "Source")
                SubElement(source, "SourceFilename").text = f
                SubElement(source, "SourceBand").text = "1"
                # SourceBand gives a 2D array (rows, cols).  We need to
                # promote it to 3D so that GDAL can place it into the
                # (stack_dim, y, x) destination.  Transpose "-1,0,1"
                # inserts a new size-1 axis at the front (the stacking
                # dimension) while keeping the original row/col order.
                SubElement(source, "SourceTranspose").text = "-1,0,1"
                # Build offset: index along stacking dim(s), then 0 for Y, X
                offset_parts = [str(i)] + ["0"] * 2
                SubElement(source, "DestSlab", offset=",".join(offset_parts))
        else:
            # No stacking dimension – single source file
            source_files = _extract_source_files(da, "")
            if source_files:
                source = SubElement(data_arr, "Source")
                SubElement(source, "SourceFilename").text = source_files[0]
                SubElement(source, "SourceBand").text = "1"
                # SourceBand gives a 2D (rows, cols) array which already
                # matches the (y, x) dimension order — identity transpose.
                #SubElement(source, "SourceTranspose").text = "0,1"
                SubElement(source, "DestSlab", offset="0,0")

        _build_array_attributes(data_arr, da)

    if validate:
        _gdal_vrt_schema().validate(vrt_root)

    return _prettify_xml(vrt_root)
