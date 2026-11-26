# Moving from the original packages

brighteyes replaces six separate packages: BrightEyes-ISM (`brighteyes_ism`),
s2ISM (`s2ism`), birfi (`birfi`), BrightEyes-FLISM (`brighteyes_flism`),
BrightEyes-MCS-Reader (`brighteyes_mcs_reader`) and BrightEyes-MCS-DataPrep
(`brighteyes_mcs_dataprep`). Code written for them keeps working after you
change its imports, because the functions and classes kept their names,
arguments and defaults. The exceptions are the few functions listed under
[Removed functions](#removed-functions).

Install the imaging extra, which also installs the MCS support:

```text
pip install brighteyes[imaging]
```

## Change the imports

Each module of BrightEyes-ISM and s2ISM now lives at one of the import paths
below. Import its names from that path exactly, for example
`from brighteyes.imaging.extraction.apr import APR`, and not from the private
files inside it, which may change.

| Original module | New import |
| --- | --- |
| `brighteyes_ism.analysis.APR_lib` | `brighteyes.imaging.extraction.apr` |
| `brighteyes_ism.analysis.FocusISM_lib` | `brighteyes.imaging.extraction.focus_ism` |
| `brighteyes_ism.analysis.Deconv_lib` | `brighteyes.imaging.extraction.deconv.ism` |
| `brighteyes_ism.analysis.FRC_lib` | `brighteyes.imaging.utils` |
| `brighteyes_ism.analysis.Tools_lib` | `brighteyes.imaging.utils` |
| `brighteyes_ism.analysis.Graph_lib` | `brighteyes.imaging.plotting` |
| `brighteyes_ism.simulation.detector` | `brighteyes.imaging.psf.sim` |
| `brighteyes_ism.simulation.PSF_sim` | `brighteyes.imaging.psf.sim` |
| `brighteyes_ism.simulation.simulator` | `brighteyes.imaging.simulator` |
| `brighteyes_ism.simulation.Tubulin_sim` | `brighteyes.imaging.simulator` |
| `brighteyes_ism.simulation.utils` (`partial_convolution`) | `brighteyes.imaging.convolution` |
| `brighteyes_ism.dataio` | `brighteyes.imaging.io` |
| `s2ism.s2ism` | `brighteyes.imaging.extraction.deconv.s2ism` |
| `s2ism.s2ism` (`partial_convolution_rfft`) | `brighteyes.imaging.convolution` |
| `s2ism.dataio` | `brighteyes.imaging.extraction.deconv.s2ism` |
| `s2ism.psf_estimator` | `brighteyes.imaging.psf.finder.psf_estimator` |
| `s2ism.mag_finder` | `brighteyes.imaging.psf.finder.mag_finder` |
| `s2ism.shift_vectors_minimizer` | `brighteyes.imaging.psf.finder.shift_vectors_minimizer` |
| `s2ism.misalignment` | `brighteyes.imaging.psf.finder.misalignment` |

The BIRFI, FLISM and MCS packages kept their module layout, so only the package
prefix changes and every module keeps its name: `birfi` becomes
`brighteyes.imaging.birfi`, `brighteyes_flism` becomes
`brighteyes.imaging.flism`, `brighteyes_mcs_reader` becomes
`brighteyes.mcs.reader` and `brighteyes_mcs_dataprep` becomes
`brighteyes.mcs.dataprep`. For example, `from birfi.birfi import Birfi` becomes
`from brighteyes.imaging.birfi.birfi import Birfi`.

A script that used the ISM and s2ISM packages changes like this:

```python
# Before
import brighteyes_ism.analysis.APR_lib as apr
import brighteyes_ism.analysis.FRC_lib as frc
import s2ism.s2ism as s2

# After
from brighteyes.imaging import utils as frc
from brighteyes.imaging.extraction import apr
from brighteyes.imaging.extraction.deconv import s2ism as s2
```

The calls that follow, such as `apr.APR(...)`, `frc.FRC(...)` and
`s2.max_likelihood_reconstruction(...)`, stay the same.

## Removed functions

These functions had no callers in the packages, their examples or their
documentation, and were removed. Code that needs one of them has to keep its
own copy. No other function or class was removed from the package versions
brighteyes was built from; functions the original packages added later are not
included.

- From `Deconv_lib`: `FRC_MultiImg_RL_FFT`, `FRC_MultiImg_RL_FFT_2`,
  `MultiImg_RL_FFT_2`, `PSF_FRC`, `convolution_matrix`, `deconv_RL_FFT`,
  `deconv_Wiener`, `deconv_Wiener_FFT`, `disk2d` and `gauss2d`. The module
  keeps `MultiImg_RL_FFT`, `PadDataset` and `UnpadDataset`.
- From `Tools_lib`: `ArgMaxND`, `FWHM`, `RadialSpectrum`, `check_saturation`,
  `fit_to_gaussian`, `low_pass`, `normalized_absolute_difference` and
  `point_cloud_from_img`.
