"""GDAL environment/configuration context manager."""

from osgeo import gdal


class Env:
    """Context manager for GDAL configuration options.

    Sets GDAL config options on entry and restores previous values on exit.
    Boolean values are converted to "YES"/"NO" strings. All other values
    are converted to strings.

    Parameters
    ----------
    **options : dict
        GDAL configuration options as keyword arguments.

    Examples
    --------
    >>> with Env(GS_NO_SIGN_REQUEST=True, GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
    ...     pass  # GDAL operations with these config options
    """

    def __init__(self, **options: object) -> None:
        self.options = options
        self._previous: dict[str, str | None] = {}

    def __enter__(self) -> "Env":
        for key, value in self.options.items():
            # Save current value (may be None if not set)
            self._previous[key] = gdal.GetConfigOption(key)
            # Convert booleans to YES/NO
            if isinstance(value, bool):
                value = "YES" if value else "NO"
            gdal.SetConfigOption(key, str(value))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for key in self.options:
            gdal.SetConfigOption(key, self._previous.get(key))
        return None
