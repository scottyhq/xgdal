"""Smoke tests for the xgdal backend."""

import numpy as np
import pyproj
import pytest
import xarray as xr
from osgeo import gdal, osr
from rasterix import RasterIndex

import xgdal


@pytest.fixture()
def tmp_tif(tmp_path):
    """Create a small single-band GeoTIFF for testing."""
    path = str(tmp_path / "test.tif")
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, 4, 3, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0.0, 1.0, 0.0, 3.0, 0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetSpatialRef(srs)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    band.WriteArray(data)
    ds.FlushCache()
    ds = None
    return path


@pytest.fixture()
def tmp_tif_with_overviews(tmp_path):
    """Create a GeoTIFF with overview levels for testing."""
    path = str(tmp_path / "test_ovr.tif")
    drv = gdal.GetDriverByName("GTiff")
    # 256x256 so we can build a couple of overviews
    nx, ny = 256, 256
    ds = drv.Create(path, nx, ny, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0.0, 1.0, 0.0, 256.0, 0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetSpatialRef(srs)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)
    data = np.arange(nx * ny, dtype=np.float32).reshape(ny, nx)
    band.WriteArray(data)
    # Build overviews at 2x and 4x reduction → levels 0 (128x128) and 1 (64x64)
    ds.BuildOverviews("NEAREST", [2, 4])
    ds.FlushCache()
    ds = None
    return path


@pytest.fixture()
def tmp_tiled_tif(tmp_path):
    """Create a tiled GeoTIFF with 64x64 block size for preferred_chunks tests."""
    path = str(tmp_path / "test_tiled.tif")
    drv = gdal.GetDriverByName("GTiff")
    nx, ny = 256, 256
    # TILED=YES with explicit 64x64 block size
    ds = drv.Create(
        path,
        nx,
        ny,
        1,
        gdal.GDT_Float32,
        options=["TILED=YES", "BLOCKXSIZE=64", "BLOCKYSIZE=64"],
    )
    ds.SetGeoTransform((0.0, 1.0, 0.0, 256.0, 0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetSpatialRef(srs)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)
    data = np.arange(nx * ny, dtype=np.float32).reshape(ny, nx)
    band.WriteArray(data)
    ds.FlushCache()
    ds = None
    return path


@pytest.fixture()
def tmp_tiled_tif_with_overviews(tmp_path):
    """Create a tiled GeoTIFF with overviews for preferred_chunks tests."""
    path = str(tmp_path / "test_tiled_ovr.tif")
    drv = gdal.GetDriverByName("GTiff")
    nx, ny = 256, 256
    ds = drv.Create(
        path,
        nx,
        ny,
        1,
        gdal.GDT_Float32,
        options=["TILED=YES", "BLOCKXSIZE=64", "BLOCKYSIZE=64"],
    )
    ds.SetGeoTransform((0.0, 1.0, 0.0, 256.0, 0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetSpatialRef(srs)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)
    data = np.arange(nx * ny, dtype=np.float32).reshape(ny, nx)
    band.WriteArray(data)
    ds.BuildOverviews("NEAREST", [2, 4])
    ds.FlushCache()
    ds = None
    return path


class TestBackendRegistration:
    def test_engine_registered(self):
        engines = xr.backends.list_engines()
        assert "xgdal" in engines

    def test_guess_can_open_tif(self):
        ep = xgdal.XGDALBackendEntrypoint()
        assert ep.guess_can_open("test.tif") is True
        assert ep.guess_can_open("test.tiff") is True
        assert ep.guess_can_open("test.vrt") is True
        assert ep.guess_can_open("test.nc") is False


class TestSingleTif:
    def test_open_dataset(self, tmp_tif):
        ds = xr.open_dataset(tmp_tif, engine="xgdal")
        assert isinstance(ds, xr.Dataset)
        assert "x" in ds.coords
        assert "y" in ds.coords
        assert "band" in ds.coords
        assert ds.sizes["x"] == 4
        assert ds.sizes["y"] == 3
        assert ds.sizes["band"] == 1

    def test_data_values(self, tmp_tif):
        ds = xr.open_dataset(tmp_tif, engine="xgdal")
        # Find the data variable (name comes from GDAL)
        data_var = list(ds.data_vars)[0]
        values = ds[data_var].values
        expected = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        np.testing.assert_array_equal(values, expected)

    def test_coords(self, tmp_tif):
        ds = xr.open_dataset(tmp_tif, engine="xgdal")
        # Pixel centres: origin_x + pixel_width/2 + i * pixel_width
        np.testing.assert_allclose(ds.x.values, [0.5, 1.5, 2.5, 3.5])
        np.testing.assert_allclose(ds.y.values, [2.5, 1.5, 0.5])

    def test_crs_via_xproj(self, tmp_tif):
        ds = xr.open_dataset(tmp_tif, engine="xgdal")
        crs = ds.proj.crs
        assert crs is not None
        assert crs == pyproj.CRS.from_epsg(4326)

    def test_drop_variables(self, tmp_tif):
        ds = xr.open_dataset(tmp_tif, engine="xgdal")
        var_name = list(ds.data_vars)[0]
        ds2 = xr.open_dataset(tmp_tif, engine="xgdal", drop_variables=[var_name])
        assert var_name not in ds2.data_vars


class TestEnv:
    def test_env_sets_and_restores(self):
        key = "XGDAL_TEST_OPTION"
        assert gdal.GetConfigOption(key) is None
        with xgdal.Env(**{key: "HELLO"}):
            assert gdal.GetConfigOption(key) == "HELLO"
        assert gdal.GetConfigOption(key) is None

    def test_env_bool_conversion(self):
        key = "XGDAL_TEST_BOOL"
        with xgdal.Env(**{key: True}):
            assert gdal.GetConfigOption(key) == "YES"
        with xgdal.Env(**{key: False}):
            assert gdal.GetConfigOption(key) == "NO"


class TestOverview:
    def test_overview_reduces_dimensions(self, tmp_tif_with_overviews):
        ds_full = xr.open_dataset(tmp_tif_with_overviews, engine="xgdal")
        ds_ovr0 = xr.open_dataset(
            tmp_tif_with_overviews,
            engine="xgdal",
            backend_kwargs={"overview_level": 0},
        )
        ds_ovr1 = xr.open_dataset(
            tmp_tif_with_overviews,
            engine="xgdal",
            backend_kwargs={"overview_level": 1},
        )
        # Full res is 256x256
        assert ds_full.sizes["x"] == 256
        assert ds_full.sizes["y"] == 256
        # Overview level 0 → 2x reduction → 128x128
        assert ds_ovr0.sizes["x"] == 128
        assert ds_ovr0.sizes["y"] == 128
        # Overview level 1 → 4x reduction → 64x64
        assert ds_ovr1.sizes["x"] == 64
        assert ds_ovr1.sizes["y"] == 64

    def test_overview_coords_cover_same_extent(self, tmp_tif_with_overviews):
        ds_full = xr.open_dataset(tmp_tif_with_overviews, engine="xgdal")
        ds_ovr = xr.open_dataset(
            tmp_tif_with_overviews,
            engine="xgdal",
            backend_kwargs={"overview_level": 0},
        )
        # Both should cover roughly the same spatial extent
        # Full: x from 0.5 to 255.5, overview: x from 1.0 to 255.0
        np.testing.assert_allclose(
            ds_full.x.values[0] - ds_full.attrs["geotransform"][1] / 2,
            ds_ovr.x.values[0] - ds_ovr.attrs["geotransform"][1] / 2,
        )

    def test_overview_data_is_readable(self, tmp_tif_with_overviews):
        ds = xr.open_dataset(
            tmp_tif_with_overviews,
            engine="xgdal",
            backend_kwargs={"overview_level": 1},
        )
        data_var = list(ds.data_vars)[0]
        values = ds[data_var].values
        assert values.shape == (1, 64, 64)
        assert not np.all(np.isnan(values))

    def test_overview_invalid_level_raises(self, tmp_tif_with_overviews):
        with pytest.raises(ValueError, match="overview_level=99 is out of range"):
            xr.open_dataset(
                tmp_tif_with_overviews,
                engine="xgdal",
                backend_kwargs={"overview_level": 99},
            )

    def test_overview_geotransform_is_scaled(self, tmp_tif_with_overviews):
        ds_full = xr.open_dataset(tmp_tif_with_overviews, engine="xgdal")
        ds_ovr = xr.open_dataset(
            tmp_tif_with_overviews,
            engine="xgdal",
            backend_kwargs={"overview_level": 0},
        )
        # Pixel width should be 2x larger for overview level 0
        full_pw = ds_full.attrs["geotransform"][1]
        ovr_pw = ds_ovr.attrs["geotransform"][1]
        np.testing.assert_allclose(ovr_pw, full_pw * 2)


class TestMaskAndScale:
    def test_fillvalue_in_attrs_by_default(self, tmp_tif):
        """Without mask_and_scale, _FillValue lives in attrs (raw metadata)."""
        ds = xr.open_dataset(tmp_tif, engine="xgdal")
        var = list(ds.data_vars.values())[0]
        assert "_FillValue" in var.attrs
        assert var.attrs["_FillValue"] == -9999.0

    def test_fillvalue_moves_to_encoding_after_decode(self, tmp_tif):
        """With mask_and_scale=True, decode_cf moves _FillValue to encoding."""
        ds = xr.open_dataset(tmp_tif, engine="xgdal", mask_and_scale=True)
        var = list(ds.data_vars.values())[0]
        assert "_FillValue" not in var.attrs
        assert "_FillValue" in var.encoding
        assert var.encoding["_FillValue"] == -9999.0

    def test_mask_and_scale_masks_nodata(self, tmp_tif):
        """mask_and_scale=True should replace nodata with NaN."""
        # Write a pixel with the nodata value
        from osgeo import gdal as _gdal

        _ds = _gdal.Open(tmp_tif, _gdal.GA_Update)
        band = _ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        arr[0, 0] = -9999.0  # set one pixel to nodata
        band.WriteArray(arr)
        _ds.FlushCache()
        _ds = None

        ds = xr.open_dataset(tmp_tif, engine="xgdal", mask_and_scale=True)
        data_var = list(ds.data_vars)[0]
        values = ds[data_var].values
        assert np.isnan(values[0, 0, 0]), "nodata pixel should be NaN after masking"
        # The encoding should still carry the original nodata
        assert ds[data_var].encoding.get("_FillValue") == -9999.0

    def test_mask_and_scale_false_preserves_raw(self, tmp_tif):
        """mask_and_scale=False should keep nodata as the raw numeric value."""
        from osgeo import gdal as _gdal

        _ds = _gdal.Open(tmp_tif, _gdal.GA_Update)
        band = _ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        arr[0, 0] = -9999.0
        band.WriteArray(arr)
        _ds.FlushCache()
        _ds = None

        ds = xr.open_dataset(tmp_tif, engine="xgdal", mask_and_scale=False)
        data_var = list(ds.data_vars)[0]
        values = ds[data_var].values
        assert values[0, 0, 0] == -9999.0, "raw nodata value should be preserved"

    def test_mdim_vrt_mask_and_scale_applies_scale_offset(self, tmp_path):
        """Reading a mdim VRT with scale_factor/add_offset Attributes applies them."""
        # Create a small multidim Zarr as the backing data store
        nx, ny = 4, 3
        zarr_path = str(tmp_path / "test.zarr")
        drv = gdal.GetDriverByName("Zarr")
        md_ds = drv.CreateMultiDimensional(zarr_path)
        rg = md_ds.GetRootGroup()

        depth_dim = rg.CreateDimension("depth", "", "", 2)
        y_dim = rg.CreateDimension("Y", "HORIZONTAL_Y", "NORTH", ny)
        x_dim = rg.CreateDimension("X", "HORIZONTAL_X", "EAST", nx)

        dt_uint16 = gdal.ExtendedDataType.Create(gdal.GDT_UInt16)
        dt_float64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        dt_int64 = gdal.ExtendedDataType.Create(gdal.GDT_Int64)

        rg.CreateMDArray("depth", [depth_dim], dt_int64).Write(
            np.array([0, 100], dtype=np.int64)
        )
        rg.CreateMDArray("Y", [y_dim], dt_float64).Write(
            np.array([2.5, 1.5, 0.5])
        )
        rg.CreateMDArray("X", [x_dim], dt_float64).Write(
            np.array([0.5, 1.5, 2.5, 3.5])
        )

        data_arr = rg.CreateMDArray("cec7", [depth_dim, y_dim, x_dim], dt_uint16)
        data = np.array([
            np.arange(nx * ny, dtype=np.uint16).reshape(ny, nx),
            (np.arange(nx * ny, dtype=np.uint16) + 100).reshape(ny, nx),
        ])
        data_arr.Write(data)
        data_arr.SetNoDataValue(65535)
        md_ds = None

        # Generate a multidim VRT via GDAL, then inject <Attribute> elements
        vrt_path = str(tmp_path / "scale_offset.vrt")
        gdal.MultiDimTranslate(vrt_path, zarr_path, format="VRT")

        import pathlib

        vrt_xml = pathlib.Path(vrt_path).read_text()
        # Insert CF attributes after <NoDataValue> inside the cec7 array
        attr_block = (
            '      <Attribute name="scale_factor">\n'
            "        <DataType>Float64</DataType>\n"
            "        <Value>0.1</Value>\n"
            "      </Attribute>\n"
            '      <Attribute name="add_offset">\n'
            "        <DataType>Float64</DataType>\n"
            "        <Value>5.0</Value>\n"
            "      </Attribute>\n"
            '      <Attribute name="units">\n'
            "        <DataType>String</DataType>\n"
            "        <Value>meq/100g</Value>\n"
            "      </Attribute>\n"
        )
        vrt_xml = vrt_xml.replace(
            "<NoDataValue>65535</NoDataValue>",
            "<NoDataValue>65535</NoDataValue>\n" + attr_block,
        )
        pathlib.Path(vrt_path).write_text(vrt_xml)

        scale_factor = 0.1
        add_offset = 5.0

        # ── Read WITHOUT mask_and_scale: attrs should carry raw metadata ──
        ds_raw = xr.open_dataset(vrt_path, engine="xgdal", mask_and_scale=False)
        raw_var = ds_raw["cec7"]
        assert raw_var.attrs["scale_factor"] == scale_factor
        assert raw_var.attrs["add_offset"] == add_offset
        assert raw_var.attrs["units"] == "meq/100g"
        raw_values = raw_var.values  # shape (2, 3, 4)

        # ── Read WITH mask_and_scale: values should be decoded ──
        ds_scaled = xr.open_dataset(vrt_path, engine="xgdal", mask_and_scale=True)
        scaled_var = ds_scaled["cec7"]

        # scale_factor / add_offset should move to encoding after decode_cf
        assert "scale_factor" not in scaled_var.attrs
        assert "add_offset" not in scaled_var.attrs
        assert scaled_var.encoding.get("scale_factor") == scale_factor
        assert scaled_var.encoding.get("add_offset") == add_offset
        # units should remain in attrs
        assert scaled_var.attrs["units"] == "meq/100g"

        # Verify decoded values: decoded = raw * scale_factor + add_offset
        expected = raw_values * scale_factor + add_offset
        np.testing.assert_allclose(scaled_var.values, expected)


class TestAccessor:
    def test_encoded_nodata_property(self, tmp_tif):
        da = xr.open_dataarray(tmp_tif, engine="xgdal")
        assert da.xgdal.encoded_nodata == -9999.0

    def test_to_raster_roundtrip(self, tmp_tif, tmp_path):
        """Write a DataArray to GeoTIFF and re-read to verify."""
        da = xr.open_dataarray(tmp_tif, engine="xgdal")
        out_path = str(tmp_path / "out.tif")
        da.xgdal.to_raster(out_path)

        # Re-read
        da2 = xr.open_dataarray(out_path, engine="xgdal")
        np.testing.assert_array_equal(da.values, da2.values)
        assert da2.xgdal.encoded_nodata == -9999.0

    def test_to_raster_preserves_nodata_after_masking(self, tmp_tif, tmp_path):
        """After mask_and_scale, to_raster should restore the original nodata."""
        from osgeo import gdal as _gdal

        # Put a nodata pixel in the source
        _ds = _gdal.Open(tmp_tif, _gdal.GA_Update)
        band = _ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        arr[0, 0] = -9999.0
        band.WriteArray(arr)
        _ds.FlushCache()
        _ds = None

        # Open with masking – nodata becomes NaN
        da = xr.open_dataarray(tmp_tif, engine="xgdal", mask_and_scale=True)
        assert np.isnan(da.values[0, 0, 0])

        # Write out
        out_path = str(tmp_path / "out_masked.tif")
        da.xgdal.to_raster(out_path)

        # Re-read raw (no masking) – the nodata value should be -9999
        da2 = xr.open_dataarray(out_path, engine="xgdal", mask_and_scale=False)
        assert da2.values[0, 0, 0] == -9999.0
        assert da2.xgdal.encoded_nodata == -9999.0

    def test_to_raster_preserves_geotransform(self, tmp_tif, tmp_path):
        da = xr.open_dataarray(tmp_tif, engine="xgdal")
        out_path = str(tmp_path / "out_gt.tif")
        da.xgdal.to_raster(out_path)

        da2 = xr.open_dataarray(out_path, engine="xgdal")
        np.testing.assert_allclose(da.coords["x"].values, da2.coords["x"].values)
        np.testing.assert_allclose(da.coords["y"].values, da2.coords["y"].values)

    def test_to_raster_preserves_crs(self, tmp_tif, tmp_path):
        da = xr.open_dataarray(tmp_tif, engine="xgdal")
        out_path = str(tmp_path / "out_crs.tif")
        da.xgdal.to_raster(out_path)

        da2 = xr.open_dataarray(out_path, engine="xgdal")
        crs = da2.proj.crs
        assert crs is not None
        assert crs == pyproj.CRS.from_epsg(4326)


class TestRasterIndex:
    def test_raster_index_assigned(self, tmp_tif):
        """Opening a GeoTIFF should produce a Dataset with a RasterIndex."""
        ds = xr.open_dataset(tmp_tif, engine="xgdal")
        raster_indexes = [
            idx for idx in ds.xindexes.values() if isinstance(idx, RasterIndex)
        ]
        assert len(raster_indexes) > 0, "Expected at least one RasterIndex"

    def test_raster_index_transform_matches_geotransform(self, tmp_tif):
        """The RasterIndex transform should match the file's geotransform."""
        ds = xr.open_dataset(tmp_tif, engine="xgdal")
        idx = next(idx for idx in ds.xindexes.values() if isinstance(idx, RasterIndex))
        # The source GeoTIFF has geotransform (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)
        gt_str = idx.as_geotransform()
        gt = tuple(float(v) for v in gt_str.split())
        assert gt == (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)

    def test_transform_updates_on_isel(self, tmp_tif):
        """Subsetting with .isel() should update the RasterIndex transform."""
        ds = xr.open_dataset(tmp_tif, engine="xgdal")
        # Original geotransform: origin_x=0.0, pixel_width=1.0,
        #                        origin_y=3.0, pixel_height=-1.0
        # Subset: skip first 2 columns, first 1 row
        subset = ds.isel(x=slice(2, None), y=slice(1, None))

        idx = next(
            idx for idx in subset.xindexes.values() if isinstance(idx, RasterIndex)
        )
        gt_str = idx.as_geotransform()
        gt = tuple(float(v) for v in gt_str.split())

        # After slicing x by 2 pixels (pixel_width=1.0): origin_x shifts by 2.0
        # After slicing y by 1 pixel (pixel_height=-1.0): origin_y shifts by -1.0
        expected_gt = (2.0, 1.0, 0.0, 2.0, 0.0, -1.0)
        np.testing.assert_allclose(gt, expected_gt)

    def test_to_raster_subset_has_updated_geotransform(self, tmp_tif, tmp_path):
        """Writing a subset to raster should use the updated geotransform."""
        da = xr.open_dataarray(tmp_tif, engine="xgdal")
        # Subset: skip first 2 columns, keep all rows
        subset = da.isel(x=slice(2, None))
        out_path = str(tmp_path / "subset.tif")
        subset.xgdal.to_raster(out_path)

        # Re-read the output file and check its geotransform
        out_ds = gdal.OpenEx(out_path, gdal.OF_RASTER)
        out_gt = out_ds.GetGeoTransform()
        out_ds = None

        # Origin x should have shifted by 2 pixels (pixel_width=1.0)
        np.testing.assert_allclose(out_gt[0], 2.0)  # origin_x
        np.testing.assert_allclose(out_gt[1], 1.0)  # pixel_width
        np.testing.assert_allclose(out_gt[5], -1.0)  # pixel_height

    def test_to_raster_subset_roundtrip(self, tmp_tif, tmp_path):
        """Subset → to_raster → re-open should give consistent coordinates."""
        da = xr.open_dataarray(tmp_tif, engine="xgdal")
        # Subset: skip first column and first row
        subset = da.isel(x=slice(1, None), y=slice(1, None))
        out_path = str(tmp_path / "subset_rt.tif")
        subset.xgdal.to_raster(out_path)

        da2 = xr.open_dataarray(out_path, engine="xgdal")
        # The re-opened file should have the same x/y coords as the subset
        np.testing.assert_allclose(subset.coords["x"].values, da2.coords["x"].values)
        np.testing.assert_allclose(subset.coords["y"].values, da2.coords["y"].values)
        np.testing.assert_array_equal(subset.values, da2.values)


# ---------------------------------------------------------------------------
# URI / VSI path normalisation
# ---------------------------------------------------------------------------


class TestURINormalisation:
    """Tests for _to_vsi_path and _to_cloud_uri helpers."""

    @pytest.mark.parametrize(
        "uri, expected",
        [
            ("gs://bucket/key.tif", "/vsigs/bucket/key.tif"),
            ("s3://bucket/key.tif", "/vsis3/bucket/key.tif"),
            ("az://container/blob.tif", "/vsiaz/container/blob.tif"),
            ("https://example.com/data.tif", "/vsicurl/https://example.com/data.tif"),
            ("http://example.com/data.tif", "/vsicurl/http://example.com/data.tif"),
            ("/vsigs/bucket/key.tif", "/vsigs/bucket/key.tif"),  # already VSI
            ("/local/path.tif", "/local/path.tif"),  # local path unchanged
        ],
    )
    def test_to_vsi_path(self, uri, expected):
        from xgdal.backend import _to_vsi_path

        assert _to_vsi_path(uri) == expected

    @pytest.mark.parametrize(
        "vsi_path, expected",
        [
            ("/vsigs/bucket/key.tif", "gs://bucket/key.tif"),
            ("/vsis3/bucket/key.tif", "s3://bucket/key.tif"),
            ("/vsiaz/container/blob.tif", "az://container/blob.tif"),
            ("/vsicurl/https://example.com/data.tif", "https://example.com/data.tif"),
            ("gs://bucket/key.tif", "gs://bucket/key.tif"),  # already cloud URI
            ("/local/path.tif", "/local/path.tif"),  # local path unchanged
        ],
    )
    def test_to_cloud_uri(self, vsi_path, expected):
        from xgdal.backend import _to_cloud_uri

        assert _to_cloud_uri(vsi_path) == expected


class TestPreferredChunks:
    """Tests for preferred_chunks encoding (on-disk block sizes)."""

    def test_untiled_tif_preferred_chunks(self, tmp_tif):
        """An untiled (stripped) GeoTIFF should still report preferred_chunks.

        GDAL reports strip-based block sizes for untiled TIFFs, typically
        (width, strip_height). The backend should expose whatever GDAL returns.
        """
        ds = xr.open_dataset(tmp_tif, engine="xgdal")
        var = list(ds.data_vars.values())[0]
        assert "preferred_chunks" in var.encoding
        pc = var.encoding["preferred_chunks"]
        assert "x" in pc
        assert "y" in pc
        # For a small untiled TIFF, GDAL typically uses full-width strips
        assert pc["x"] == 4  # full width of the raster

    def test_tiled_tif_preferred_chunks(self, tmp_tiled_tif):
        """A tiled GeoTIFF with 64x64 tiles should report 64x64 preferred_chunks."""
        ds = xr.open_dataset(tmp_tiled_tif, engine="xgdal")
        var = list(ds.data_vars.values())[0]
        assert "preferred_chunks" in var.encoding
        pc = var.encoding["preferred_chunks"]
        assert pc["x"] == 64
        assert pc["y"] == 64

    def test_tiled_tif_preferred_chunks_band(self, tmp_tiled_tif):
        """For a single-band file, band is not in preferred_chunks (n_bands <= 1)."""
        ds = xr.open_dataset(tmp_tiled_tif, engine="xgdal")
        var = list(ds.data_vars.values())[0]
        pc = var.encoding["preferred_chunks"]
        # Single-band files omit the band key (only included when n_bands > 1)
        assert "band" not in pc

    def test_tiled_tif_chunks_empty_dict(self, tmp_tiled_tif):
        """chunks={} should use the on-disk block sizes from preferred_chunks."""
        da = xr.open_dataarray(tmp_tiled_tif, engine="xgdal", chunks={})
        # 256x256 raster with 64x64 tiles → 4 chunks per dimension
        assert da.chunks is not None
        # y chunks: (64, 64, 64, 64)
        assert all(c == 64 for c in da.chunks[1])  # y dim (index 1 because band=0)
        assert len(da.chunks[1]) == 4
        # x chunks: (64, 64, 64, 64)
        assert all(c == 64 for c in da.chunks[2])  # x dim
        assert len(da.chunks[2]) == 4

    def test_tiled_tif_chunks_partial_override(self, tmp_tiled_tif):
        """chunks={'x': 128} should override x but use preferred for y."""
        da = xr.open_dataarray(tmp_tiled_tif, engine="xgdal", chunks={"x": 128})
        assert da.chunks is not None
        # x should be overridden to 128
        assert all(c == 128 for c in da.chunks[2])
        # y should still use preferred 64
        assert all(c == 64 for c in da.chunks[1])

    def test_overview_preferred_chunks(self, tmp_tiled_tif_with_overviews):
        """Overview level should report its own block sizes."""
        ds = xr.open_dataset(
            tmp_tiled_tif_with_overviews,
            engine="xgdal",
            backend_kwargs={"overview_level": 0},
        )
        var = list(ds.data_vars.values())[0]
        assert "preferred_chunks" in var.encoding
        # Overview block sizes may differ from the base resolution but
        # should still be present and positive
        pc = var.encoding["preferred_chunks"]
        assert pc["x"] > 0
        assert pc["y"] > 0
