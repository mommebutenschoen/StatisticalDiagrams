"""
Target Diagrams using robust statistics:

    * average: median
    * scale: inter-quartile arange
    * scale difference: median absolute error
    * correlation: spearman

Example:

The following example plots a target diagram based on robust statistics for
three randomly generated datasets with respect to randomly distributed reference
data: one uncorrelated to the reference data with no added bias, one weakly
correlated with substantially smaller scale and added negative bias and one
strongly correlated one with substantially larger scale and added positive bias.
Two plots are generated, one using the ``TargetStatistics`` class that
internally computes the summary statistics from the datasets and one using the
``TargetDiagram`` class and pre-computed summary statistics. The example shows
that the outcome is equivalent.

.. code-block:: python

  from numpy.random import randn
  from numpy import median
  from scipy.stats import spearmanr
  from StatisticalDiagrams.robust import *
  from StatisticalDiagrams.RobustStatistics import MAE,IQR
  from matplotlib.pyplot import figure,show

  ref=randn(10)
  a=randn(10)
  b=.5*(.25*ref+.75*randn(10)) - .5
  c=2.*(.75*ref+.25*randn(10)) + .5

  f1=figure()
  TD=TargetStatistics(a,ref,precision=1.e-15)
  TD(b,ref,precision=1.e-15)
  TD(c,ref,precision=1.e-15)
  show()

  R1,p=spearmanr(a,ref)
  scale1=IQR(a)
  refscale=IQR(ref)
  E0_1=median(a-ref)/refscale
  E1=MAE(a-E0_1*refscale,ref)/refscale

  R2,p=spearmanr(b,ref)
  E0_2=median(b-ref)/refscale
  scale2=IQR(b)
  E2=MAE(b-E0_2*refscale,ref)/refscale

  R3,p=spearmanr(c,ref)
  E0_3=median(c-ref)/refscale
  scale3=IQR(c)
  E3=MAE(c-E0_3*refscale,ref)/refscale

  f2=figure()
  TD=TargetDiagram(scale1/refscale,E0_1,E1,R1)
  TD(scale2/refscale,E0_2,E2,R2)
  TD(scale3/refscale,E0_3,E3,R3)
  show()
"""

from __future__ import print_function
import logging
from numpy import sqrt,arange,sin,cos,pi,abs,arccos,array,median,atleast_1d
from scipy.stats import spearmanr
from matplotlib.pyplot import plot,axis,scatter,xlabel,ylabel,clabel,colorbar,text,subplot,tick_params,xticks,yticks
from .RobustStatistics import MAE,Qn,unbiasedMAE,IQR,Sn,MID
from matplotlib.ticker import MultipleLocator,NullFormatter

scaleFun=IQR
scaleDiff=MAE
scaleLable="IQR"
diffLable="MAE"

class StatsDiagram:

    """Base class for robust statistical summary diagrams based on two arrays of
    the same size that will be compared on a point to point base.
    The first array is considered the data to be evaluated, the second is
    the reference data. It computes all the basic metrics using the
    ``_stats`` function:

    Attributes:
          scale (float): scale of reference data
          E0 (float): the median bias of the two data sets, relative to scale
          gama(float): the ratio of the scales of data over reference data
          R(float): Spearman correlation
          p(float): the Spearman correlation with p-value
          E(float): the scale difference of the two dataset
          csv(string): collects the summary statistics of the instance to be
            written to csv file, when desired.
    """

    def __init__(self,data,refdata,precision,*opts,**keys):

        """Calls summary statistics function from input data with respect to
        reference data. Initialises csv attribute. In the robust case a
        precision of the reference data is required. This represents the
        measurment precision and ensures that the scale measure can not be
        lesser than the precision. In case of the latter a warning is issued
        and the scale measure of the reference is replaced by the precision.

        Args:
            data(float array): input data
            refdata(float array): references data, same shape as input data
            precision(float arrya): reference data precision
            *opts: positional arguments passed to summary statistics function
            **keys: keyword arguments passed to summary statistics function
        """

        self.csv=""
        self._stats(data,refdata,precision,*opts,**keys)

    def __call__(self,data,refdata,precision,*opts,**keys):

        """Recomputes summary statistics for new input and reference data.

        Args:
            data(float array): input data
            refdata(float array): references data, same shape as input data
            precision(float arrya): reference data precision
            *opts: positional arguments passed to summary statistics function
            **keys: keyword arguments passed to summary statistics function
        """

        self._stats(data,refdata,precision,*opts,**keys)

    def _stats(self,data,refdata,precision,*opts,**keys):

        """Summary statistics functionst that computes the relevant metrics of
        the input data to be evaluated with respect to the reference data, adds
        them to the csv attribute and prints the summary statistics to stdout.

        Args:
            data(float array): input data
            refdata(float array): references data, same shape as input data
            precision(float arrya): reference data precision
            *opts: positional arguments passed to summary statistics function
            **keys: keyword arguments passed to summary statistics function
        """

        dat=array(data).ravel()
        ref=array(refdata).ravel()
        self.scale=scaleFun(ref)
        if self.scale<precision:
            logging.warning("Reference scale lesser than measurment precision!")
            self.scale=precision
            logging.warning("\treplaced by precision",precision)
        bias=median(dat-ref)
        self.E0=(median(dat-ref))/self.scale
        self.gamma=scaleFun(dat)/self.scale
        self.R,self.p=spearmanr(dat,ref)
        self.E=scaleDiff(data-bias,ref)/self.scale
        self.addCSV()
        print(self)

    def addCSV(self,):

        """Add new set of summary statistics to csv attribute."""

        self.csv+="{:1.5f}, {:1.5f}, {:1.5f}, {:1.5f}\n".format(self.E0,self.E,self.scale,self.R)

    def writeCSV(self,filename,*opts,**keys):

        """Write csv attribute to file.

        Args:
            filename (string): filename for csv filename
            *opts: positional arguments passed to `open` function
            **args" keyword arguments passed to `open` function
        """

        with open(filename,*opts,**keys) as fid:
            fid.write("Bias, unbiased MAE, IQR, Spearman Correlation\n")
            fid.write(self.csv)

    def __str__(self):

        return "\tNormalised Bias: "+str(self.E0)+\
            "\n\tNormalised Difference Scale: "+str(self.E)+\
            "\n\tCorrelation Coefficient: "+str(self.R)+\
            "\n\t\t with p-value: "+str(self.p)+\
	    "\n\tScale Ratio (Data/Reference): "+str(self.gamma)+\
	    "\n\tReference Scale: "+str(self.scale)+'\n'

class Stats:

    """Base class for statistical summary diagrams, using precomuted metrics rather than the full datasets as input."""

    def __init__(self,gam,E0,E,rho):

        """Loading the necessary metrics.

        Args:
            gam (float): scale ratio (Data/Reference)
            E0 (float): bias
            rho (float): correlation coefficient
        """

        self.E0=E0
        self.R=rho
        self.gamma=gam
        self.E=E
        print(self)
        self.addCSV()

    def __call__(self,gam,E0,E,rho):

        """Loading new metrics.

        Args:
            gam (float): scale ratio (Data/Reference)
            E0 (float): bias
            rho (float): correlation coefficient
        """

        self.E0=E0
        self.R=rho
        self.gamma=gam
        self.E=E
        print(self)
        self.addCSV()

    def addCSV(self,):

        """Add new set of summary statistics to csv attribute."""

        self.csv+="{:1.5f}, {:1.5f}, {:1.5f}, {:1.5f}\n".format(self.E0,self.E,self.scale,self.R)

    def writeCSV(self,filename,*opts,**keys):

        """Write csv attribute to file.

        Args:
            filename (string): filename for csv filename
            *opts: positional arguments passed to `open` function
            **args" keyword arguments passed to `open` function
        """

        with open(filename,*opts,**keys) as fid:
            fid.write("Bias, unbiased MAE, IQR,Spearman Correlation\n")
            fid.write(self.csv)

    def __str__(self):

        return "\tNormalised Bias: "+str(self.E0)+\
            "\n\tNormalised Difference Scale: "+str(self.E)+\
            "\n\tCorrelation Coefficient: "+str(self.R)+\
            "\n\tScale Ratio (Data/Reference): "+str(self.gamma)

class Target:

    """Base class providing a function to draw the grid for Target Diagrams."""

    def drawTargetGrid(self):

        """Draws Target Diagram grid."""

        majL=MultipleLocator(1)
        minL=MultipleLocator(.5)
        majF=NullFormatter()
        ax=subplot(111)
        plot(0.,0.,'ko',markersize=12)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_position('center')
        ax.spines['top'].set_position('center')
        tick_params(which='major',length=5,width=1.5)
        tick_params(which='minor',length=3,width=1.2)
        #ax.xaxis.set_ticklabels([])
        #ax.yaxis.set_ticklabels([])
        ax.xaxis.set_major_formatter(majF)
        ax.yaxis.set_major_formatter(majF)
        ax.xaxis.set_major_locator(majL)
        ax.xaxis.set_minor_locator(minL)
        ax.yaxis.set_major_locator(majL)
        ax.yaxis.set_minor_locator(minL)
        xticks((-1,1),("-1","1"))
        yticks((-1,1),("-1","1"))
        #radius at minimum RMSD' given R:
        #  Mr=sqrt(1+R^2-2R^2)
        #for R=0.7:
        #plot(.71414284285428498*sin(a),.71414284285428498*cos(a),'k--')
        #radius at observ. uncertainty:
        #plot(.5*sin(a),.5*cos(a),'k:')
        #plot((0,),(0,),'k+')
        text(-.05,.5,'${Bias}/{IQR}_{ref}$',fontsize=16,transform=ax.transAxes,rotation=90,verticalalignment='center',horizontalalignment='center')
        text(.5,-.05,"${sign}({IQR}-{IQR}_{ref})*{MAE'}/{IQR}_{ref}$",fontsize=16,transform=ax.transAxes,verticalalignment='center',horizontalalignment='center')


class TargetDiagram(Target,Stats):

    """Class for drawing a Target diagram from pre-calculated metrics.

    Attributes:
          scale (float): scale of reference data
          E0 (float): the median bias of the two data sets, relative to scale
          gama(float): the ratio of the scales of data over reference data
          R(float): Spearman correlation
          p(float): the Spearman correlation with p-value
          E(float): the scale difference of the two dataset
          csv(string): collects the summary statistics of the instance to be
            written to csv file, when desired.
    """

    def __init__(self,gam,E0,E,rho,marker='o',s=40,antiCorrelation=False,*opts,**keys):

        """Initialises the class given the pre-calculated metrics and draws
        the diagram grid with the first point.
        Markers in the diagram are colour-coded using the bias.

        Args:
            gam (float): scale ratio (Data/Reference)
            E0 (float): bias
            rho (float): correlation coefficient
            marker: shape used to show points, should be a hollow shape as it is
                filled with colour code for bias.
            s (integer scalar or array_like): marker size in points
            antiCorrelation (boolean): if True, show negative correlations
            opts: positional arguments passed to ``add`` function
            keys: keyword arguments passed to ``add`` function
        """

        Stats.__init__(self,gam,E0,E,rho)
        self.drawTargetGrid()
        if antiCorrelation:
          self._cmin=-1.
        else:
          self._cmin=0.
        self._cmax=1.
        self._lpos=[]
        self.add(self.gamma,self.E0,self.E,self.R,marker=marker,s=s,*opts,**keys)
        self.cbar=colorbar()
        self.cbar.set_label('Correlation Coefficient')

    def __call__(self,gam,E0,E,rho,marker='o',s=40,*opts,**keys):

        """Adds points to the diagram adjusting the colour codes

        Args:
            gam (float): scale ratio (Data/Reference)
            E0 (float): bias
            rho (float): correlation coefficient
            marker: shape used to show points, should be a hollow shape as it is
                filled with colour code for bias.
            s (integer scalar or array_like): marker size in points
            opts: positional arguments passed to ``add`` function
            keys: keyword arguments passed to ``add`` function
        """

        Stats.__call__(self,gam,E0,E,rho)
        self.add(self.gamma,self.E0,self.E,self.R,marker=marker,s=s,*opts,**keys)

    def add(self,gam,E0,E,R,marker='o',s=40,*opts,**keys):

        """Function to add additional points to the diagram, usually invoked
        by means of the ``__call__`` function.

        Args:
            gam (float): scale ratio (Data/Reference)
            E0 (float): bias
            rho (float): correlation coefficient
            marker: shape used to show points, should be a hollow shape as it is
                filled with colour code for bias.
            s (integer scalar or array_like): marker size in points
            opts: positional arguments passed to ``add`` function
            keys: keyword arguments passed to ``add`` function
        """

        sig= gam>1 and 1 or -1
        scatter(atleast_1d(sig*E),atleast_1d(E0),c=atleast_1d(R),
            vmin=self._cmin,vmax=self._cmax,marker=marker,s=s,*opts,**keys)
        self._lpos.append((sig*E,E0))
        rmax=max(abs(array(axis('scaled'))).max(),1.5)
        plot((0,0),(-rmax,rmax),'k-')
        plot((rmax,-rmax),(0,0),'k-')
        axis(xmin=-rmax,xmax=rmax,ymax=rmax,ymin=-rmax)

    def labels(self,lstr,*opts,**keys):

        """Adds labels ``lstr``` to the points in the diagram.

        Args:
            lstr (string): string for lable
            *opts: positional arguments passed to ``matplotlib.pyplot.text``
                function.
            **keys: keyword arguments passed to ``matplotlib.pyplot.text``
                function.
        """

        rmax=max(abs(array(axis())).max(),1.5)
        for n,p in enumerate(self._lpos):
            text(p[0]+.025*rmax,p[1]+.025*rmax,lstr[n],*opts,**keys)

class TargetStatistics(StatsDiagram,TargetDiagram):

    """Class for drawing a Target diagram from input and reference data.

    Attributes:
          scale (float): scale of reference data
          E0 (float): the median bias of the two data sets, relative to scale
          gama(float): the ratio of the scales of data over reference data
          R(float): Spearman correlation
          p(float): the Spearman correlation with p-value
          E(float): the scale difference of the two dataset
          csv(string): collects the summary statistics of the instance to be
            written to csv file, when desired.
          cbar(matplotlib.colors.colorbar): colorbar of target plot
    """

    def __init__(self,data,refdata,precision,marker='o',s=40,antiCorrelation=False,*opts,**keys):

        """Calls summary statistics function from input data with respect to
        reference data. Initialises csv attribute. In the robust case a
        precision of the reference data is required. This represents the
        measurment precision and ensures that the scale measure can not be
        lesser than the precision. In case of the latter a warning is issued
        and the scale measure of the reference is replaced by the precision.

        Args:
            data(float array): input data
            refdata(float array): references data, same shape as input data
            precision(float arrya): reference data precision
            marker: shape used to show points, should be a hollow shape as it is
                filled with colour code for bias.
            s (integer scalar or array_like): marker size in points
            antiCorrelation (boolean): if True, show negative correlations
            *opts: positional arguments passed to summary statistics function
            **keys: keyword arguments passed to summary statistics function
        """

        StatsDiagram.__init__(self,data,refdata,precision,*opts,**keys)
        self.drawTargetGrid()
        if antiCorrelation:
          self._cmin=-1.
        else:
          self._cmin=0.
        self._cmax=1.
        self._lpos=[]
        self.add(self.gamma,self.E0,self.E,self.R,marker=marker,s=s,*opts,**keys)
        self.cbar=colorbar()
        self.cbar.set_label('Correlation Coefficient')

    def __call__(self,data,refdata,precision,marker='o',s=40,*opts,**keys):

        """Adds points to the diagram adjusting the colour codes.

        Args:
            data(float array): input data
            refdata(float array): references data, same shape as input data
            precision(float arrya): reference data precision
            marker: shape used to show points, should be a hollow shape as it is
                filled with colour code for bias.
            s (integer scalar or array_like): marker size in points
            *opts: positional arguments passed to summary statistics function
            **keys: keyword arguments passed to summary statistics function
        """

        StatsDiagram.__call__(self,data,refdata,precision,*opts,**keys)
        self.add(self.gamma,self.E0,self.E,self.R,marker=marker,s=s,*opts,**keys)

    def add(self,gam,E0,E,R,marker='o',s=40,*opts,**keys):

        """Function to add additional points to the diagram, usually invoked
        by means of the ``__call__`` function.

        Args:
            gam (float): scale ratio (Data/Reference)
            E0 (float): bias
            rho (float): correlation coefficient
            marker: shape used to show points, should be a hollow shape as it is
                filled with colour code for bias.
            s (integer scalar or array_like): marker size in points
            opts: positional arguments passed to ``add`` function
            keys: keyword arguments passed to ``add`` function
        """

        sig= gam>1 and 1 or -1
        scatter(atleast_1d(sig*E),atleast_1d(E0),c=atleast_1d(R),
            vmin=self._cmin,vmax=self._cmax,marker=marker,s=s,*opts,**keys)
        self._lpos.append((sig*E,E0))
        rmax=max(abs(array(axis('scaled'))).max(),1.5)
        axis(xmin=-rmax,xmax=rmax,ymax=rmax,ymin=-rmax)

    def labels(self,lstr,*opts,**keys):

        """Adds labels ``lstr``` to the points in the diagram.

        Args:
            lstr (string): string for lable
            *opts: positional arguments passed to ``matplotlib.pyplot.text``
                function.
            **keys: keyword arguments passed to ``matplotlib.pyplot.text``
                function.
        """

        rmax=max(abs(array(axis())).max(),1.5)
        for n,p in enumerate(self._lpos):
            text(p[0]+.025*rmax,p[1]+.025*rmax,lstr[n],*opts,**keys)
