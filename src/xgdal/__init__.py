"""xgdal: An Xarray backend using GDAL bindings."""

from xgdal.backend import XGDALBackendEntrypoint, _to_cloud_uri, _to_vsi_path
from xgdal.env import Env

# Import accessor module to register the ``da.xgdal`` accessor
import xgdal.accessor  # noqa: F401

__all__ = [
    "Env",
    "XGDALBackendEntrypoint",
    "_to_cloud_uri",
    "_to_vsi_path",
]
