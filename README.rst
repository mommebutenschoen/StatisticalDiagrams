===================
StatisticalDiagrams
===================

Python package for drawing statistical summary diagrams such as Taylor or Target Diagrams.


Installation:
-------------

After downloading the source from github_ install via pip, descending
into the top-level of the source tree
and launching::

  pip install .

or to install in developers mode::

  pip install -e .

Or install the latest release from PyPI::

  pip install StatisticalDiagrams

.. _github: https://github.com/mommebutenschoen/StatisticalDiagrams


Documentation
-------------

Documentation of this package can be found on readthedocs_.

.. _readthedocs: https://statisticaldiagrams.readthedocs.io/


Simple Example:
---------------

.. code-block:: python

  from StatsDiagram import *
  from numpy.random import randn
  from matplotlib.pyplot import show,subplot
  from scipy.stats import pearsonr

  a=randn(10)
  b=randn(10)
  ref=randn(10)
  subplot(221)
  TD=TargetStatistics(a,ref)
  TD(b,ref)
  subplot(222)
  TD=TaylorStatistics(a,ref)
  TD(b,ref)

  std1=a.std()
  std2=b.std()
  refstd=ref.std()
  R1,p=pearsonr(a,ref)
  E1=(a.mean()-ref.mean())/refstd
  G1=std1/refstd
  R2,p=pearsonr(b,ref)
  E2=(b.mean()-ref.mean())/refstd
  G2=std2/refstd

  subplot(223)
  TayD=TargetDiagram(G1,E1,R1,)
  TayD(G2,E2,R2,)
  subplot(224)
  TarD=TaylorDiagram(G1,E1,R1,)
  TarD(G2,E2,R2,)

  show()
