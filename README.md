# brighteyes

> [!CAUTION]
> Currently work in progress, unstable. Do not use yet.

brighteyes is a Python toolbox for image scanning microscopy (ISM) and fluorescence lifetime ISM (FLISM) with array detectors. It includes the MCS file readers and calibration tools used by the imaging workflows.

```
pip install brighteyes[imaging]
```

Install the extra of each subpackage you need:

| Subpackage | Content | Extra |
|---|---|---|
| `brighteyes.mcs` | Reading and preparing HDF5 files from the MCS acquisition software | `mcs` |
| `brighteyes.imaging` | ISM, FLISM, s2ISM and birfi; also installs `mcs` | `imaging` |

Code written for the original BrightEyes-ISM, s2ISM, birfi, BrightEyes-FLISM and MCS packages needs new imports; [Moving from the original packages](docs/migration.md) lists them.

## Development

Install the imaging and development dependencies with `uv sync --locked --extra imaging --group dev`. Build the pure Python package with `uv build`.
