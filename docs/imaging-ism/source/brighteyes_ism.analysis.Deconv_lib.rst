.. Origin: VicidominiLab/BrightEyes-ISM, docs/source/brighteyes_ism.analysis.Deconv_lib.rst at d799720.

ISM deconvolution
=================

Replace imports from ``brighteyes.imaging.analysis.Deconv_lib`` with imports
from ``brighteyes.imaging.extraction.deconv.ism``. Keep the same arguments
when calling the functions.

.. code-block:: python

   from brighteyes.imaging.extraction.deconv.ism import MultiImg_RL_FFT

   reconstruction = MultiImg_RL_FFT(psf, image, max_iter=5)

.. automodule:: brighteyes.imaging.extraction.deconv.ism
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
