"""
Standard versions of statistical summary diagrams (Taylor and Target).

Examples:

The following example plots a Taylor and Target Diagrams for three randomly
generated datasets with respect to randomly distributed reference data: one
uncorrelated to the reference data with no added bias, one weakly correlated
with substantially smaller scale and added negative bias and one strongly
correlated one with substantially larger scale and added positive bias.
Four plots are generated, two using the ``TargetStatistics`` and
``TaylorStatistics`` classes that internally computes the summary statistics
from the datasets and two using the ``TargetDiagram``  and ``TaylorDiagram``
class and pre-computed summary statistics. The example shows that the outcome is
equivalent.

.. code-block:: python

  from StatisticalDiagrams import *
  from numpy.random import randn
  from matplotlib.pyplot import figure,show

  ref=randn(10)
  a=randn(10)
  b=.5*(.25*ref+.75*randn(10)) - .5
  c=2.*(.75*ref+.25*randn(10)) + .5

  f=figure(figsize=[12,10])
  ax1=f.add_subplot(221)
  TarS=TargetStatistics(a,ref,ax=ax1)
  TarS(b,ref)
  TarS(c,ref)
  ax2=f.add_subplot(222)
  TayS=TaylorStatistics(a,ref,ax=ax2)
  TayS(b,ref)
  TayS(c,ref)

  std1=a.std()
  std2=b.std()
  std3=c.std()
  refstd=ref.std()
  R1,p=pearsonr(a,ref)
  E1=(a.mean()-ref.mean())/refstd
  G1=std1/refstd
  R2,p=pearsonr(b,ref)
  E2=(b.mean()-ref.mean())/refstd
  G2=std2/refstd
  R3,p=pearsonr(c,ref)
  E3=(c.mean()-ref.mean())/refstd
  G3=std3/refstd

  ax3=f.add_subplot(223)
  TarD=TargetDiagram(G1,E1,R1,ax=ax3)
  TarD(G2,E2,R2,)
  TarD(G3,E3,R3,)
  ax4=f.add_subplot(224)
  TayD=TaylorDiagram(G1,E1,R1,ax=ax4)
  TayD(G2,E2,R2,)
  TayD(G3,E3,R3,)

  show()

"""
from __future__ import print_function
from numpy import sqrt,arange,sin,cos,pi,abs,arccos,array,atleast_1d
from scipy.stats import pearsonr
from matplotlib.pyplot import figure

def rmsds(gamma,R):

    """Computes normalised unbiased root-mean-square difference from correlatio
    and standard deviation ratio.

    Args:
        gamma (float): ratio of standard deviations
        R (float): correlation coefficient

    Returns:
        normalised unbiased root-mean-square difference.
    """

    return sqrt(1.+gamma**2-2.*gamma*R)

class StatsDiagram:

    """Base class for statistical summary diagrams based on two arrays of
    the same size that will be compared on a point to point base.
    The first array is considered the data to be evaluated, the second is
    the reference data. It computes all the basic metrics using the
    ``_stats`` function:

    Attributes:
          std (float): the standard deviation of the reference data
          E0 (float): the mean normalised bias of the two data sets
          gamma (float): the ratio of the std of data over reference data
          R (float): Pearson correlation
          p (float): p-value of Pearson correlation
          E (float): the normalised root-mean square difference of the two
            dataset
          csv (string): collects the summary statistics of the instance to be
            written to csv file, when desired.
    """

    def __init__(self,data,refdata,*opts,**keys):

        """Calls summary statistics function from input data with respect to
        reference data. Initialises csv attribute.

        Args:
            data (float array): input data
            refdata (float array): references data, same shape as input data
            *opts: positional arguments passed to summary statistics function
            **keys: keyword arguments passed to summary statistics function
        """

        self.csv=""
        self._stats(data,refdata,*opts,**keys)

    def __call__(self,data,refdata,*opts,**keys):

        """Recomputes summary statistics for new input and reference data.

        Args:
            data (float array): input data
            refdata (float array): references data, same shape as input data
            *opts: positional arguments passed to summary statistics function
            **keys: keyword arguments passed to summary statistics function
        """

        self._stats(data,refdata,*opts,**keys)

    def _stats(self,data,refdata,*opts,**keys):

        """Summary statistics functionst that computes the relevant metrics of
        the input data to be evaluated with respect to the reference data, adds
        them to the csv attribute and prints the summary statistics to stdout.

        Args:
            data (float array): input data
            refdata (float array): references data, same shape as input data
            *opts: positional arguments passed to summary statistics function
            **keys: keyword arguments passed to summary statistics function
        """

        dat=array(data).ravel()
        ref=array(refdata).ravel()
        self.std=ref.std()
        self.E0=(dat-ref).mean()/self.std
        self.gamma=dat.std()/self.std
        self.R,self.p=pearsonr(dat,ref)
        self.E=rmsds(self.gamma,self.R)
        self.addCSV()
        print(self)

    def addCSV(self,):

        """Add new set of summary statistics to csv attribute."""

        self.csv+="{:1.5f}, {:1.5f}, {:1.5f}, {:1.5f}\n".format(self.E0,self.E,self.std,self.R)

    def writeCSV(self,filename,*opts,**keys):

        """Write csv attribute to file.

        Args:
            filename (string): filename for csv filename
            *opts: positional arguments passed to `open` function
            **args" keyword arguments passed to `open` function
        """

        with open(filename,*opts,**keys) as fid:
            fid.write("Bias, unbiased RMS, STD, Pearson Correlation\n")
            fid.write(self.csv)

    def __str__(self):

        return "\tNormalised Bias: "+str(self.E0)+\
           "\n\tNormalised Unbiased RMSD: "+str(self.E)+\
           "\n\tNormalised RMSD: "+str(sqrt(self.E0**2+self.E**2))+\
           "\n\tCorrelation Coefficient: "+str(self.R)+\
           "\n\t\t with p-value: "+str(self.p)+\
           "\n\tSTD Ratio (Data/Reference): "+str(self.gamma)+\
           "\n\tReference STD: "+str(self.std)+'\n'


class Stats:

    """Base class for statistical summary diagrams, using precomuted metrics rather than the full datasets as input."""

    def __init__(self,gam,E0,rho):

        """Loading the necessary metrics.

        Args:
            gam (float): STD ratio (Data/Reference)
            E0 (float): Normalised mean bias
            rho (float): Correlation Coefficient
        """
        self.E0=E0
        self.R=rho
        self.gamma=gam
        self.E=rmsds(gam,rho)
        self.csv=""
        self.addCSV()
        print(self)

    def __call__(self,gam,E0,rho):

        """Loading new metrics.

        Args:
            gam (float): STD ratio (Data/Reference)
            E0 (float): Mean bias
            rho (float): Correlation Coefficient
        """

        self.E0=E0
        self.R=rho
        self.gamma=gam
        self.E=rmsds(gam,rho)
        self.addCSV()
        print(self)

    def addCSV(self,):

        """Add new set of summary statistics to csv attribute."""

        self.csv+="{:1.5f}, {:1.5f}, {:1.5f}, {:1.5f}\n".format(self.E0,self.E,self.gamma,self.R)

    def writeCSV(self,filename,*opts,**keys):

        """Write csv attribute to file.

        Args:
            filename (string): filename for csv filename
            *opts: positional arguments passed to `open` function
            **args" keyword arguments passed to `open` function
        """

        with open(filename,*opts,**keys) as fid:
            fid.write("Bias, unbiased RMS, STD, Pearson Correlation\n")
            fid.write(self.csv)

    def __str__(self):
        return "\tNormalised Bias: "+str(self.E0)+\
          "\n\tNormalised Unbiased RMSD: "+str(self.E)+\
          "\n\tNormalised RMSD: "+str(sqrt(self.E0**2+self.E**2))+\
          "\n\tCorrelation Coefficient: "+str(self.R)+\
          "\n\tSTD Ratio (Data/Reference): "+str(self.gamma)

class Target:

    """Base class providing a function to draw the grid for Target Diagrams.

    Args:
        ax (matplotlib.axes.Axes): axes containing the diagram.
    """

    def __init__(self,ax):

        """
        Args:
            ax (matplotlib.axes.Axes): axes to use, if False creates new Axes
        """

        self.ax=ax

    def drawTargetGrid(self,):

        """Draws Target Diagram grid."""

        a=arange(0,2.01*pi,.02*pi)
        self.ax.plot(sin(a),cos(a),'k')
        #radius at minimum RMSD' given R:
        #  Mr=sqrt(1+R^2-2R^2)
        #for R=0.7:
        self.ax.plot(.71414284285428498*sin(a),.71414284285428498*cos(a),'k--')
        #radius at observ. uncertainty:
        #plot(.5*sin(a),.5*cos(a),'k:')
        self.ax.plot((0,),(0,),'k+')
        self.ax.set_ylabel('${Bias}/\sigma_{ref}$',fontsize=16)
        self.ax.set_xlabel('${sign}(\sigma-\sigma_{ref})*{RMSD}\'/\sigma_{ref}$',fontsize=16)

class Taylor:

    """Base class providing a function to draw the grid for Taylor Diagrams for
    inheritance along with stats class.

    Args:
            ax (matplotlib.axes.Axes): axes containing the diagram.
    """

    def __init__(self,ax):

        """
        Args:
            ax (matplotlib.axes.Axes): axes to use, if False creates new Axes
        """

        self.ax=ax

    def drawTaylorGrid(self,R,dr,antiCorrelation):
        """Draws Taylor Diagram grid.

        Args:
            R (float): length of rays from origin in Taylor plot
            dr (float): step-size of circles around ideal point (1,0) in Taylor
                plot
            antiCorrelation (boolean): plot also anticorrelation half of diagram
        """
        if antiCorrelation:
          a0=-1.
        else:
          a0=0.
	    #Draw circles:
        a=arange(a0,1.01,.05)*.5*pi
        self.ax.plot(sin(a),cos(a),'k-')
        self.ax.plot(.5*sin(a),.5*cos(a),'k:')
        n=R/.5
        for m in arange(3,n+1):
            self.ax.plot(m*.5*sin(a),m*.5*cos(a),'k:')
        #Draw rays for correlations at .99,.75,.5,.25 steps:
        rays=list(arange(a0,1.05,.25))
        rays.append(.99)
        if a0==-1.: rays.append(-.99)
        for rho in rays:
            self.ax.plot((R*rho,0),(R*sin(arccos(rho)),0),'k:')
            d = rho>=0. and 1.02 or 1.25
            self.ax.text(d*R*rho,1.01*R*sin(arccos(rho)),str(rho),fontsize=8)
            self.ax.text(1.01*R*sin(pi*.25),1.02*R*cos(pi*.25),r'$\rho$',rotation=-45,fontsize=16)
        #text(0.,1.02*R*cos(0.),'0')
        #text(1.03*R*sin(.5*pi),0.,'1')
        goOn=True
        r=dr
        a=arange(-1.,1.01,.05)*.5*pi
        self.ax.plot((1,),(0,),'ko')
        while goOn:
            xx=[]
            yy=[]
            for p in a:
              x=r*sin(p)+1.
              y=r*cos(p)
              if antiCorrelation:
               if x**2+y**2<R**2:
                xx.append(x)
                yy.append(y)
              else:
               if x**2+y**2<R**2:
                if x>0.:
                 xx.append(x)
                 yy.append(y)
            if len(xx)>0:
                self.ax.plot(xx,yy,'k--')
            else:
                goOn=False
            r+=dr
        self.ax.set_xlabel('$\sigma/\sigma_{ref}$',fontsize=16)
        self.ax.set_ylabel('$\sigma/\sigma_{ref}$',fontsize=16)

class TaylorDiagram(Taylor,Stats):

    """Class for drawing a Taylor diagram from pre-calculated metrics.

    Attributes:
          std (float): the standard deviation of the reference data
          E0 (float): normalised mean bias of the two data sets
          gama(float): the ratio of the std of data over reference data
          R(float): Pearson correlation
          p(float): p-value of Pearson correlation
          E(float): normalised root-mean square difference of the two dataset
          csv(string): collects the summary statistics of the instance to be
            written to csv file, when desired.
          ax (matplotlib.axes.Axes): axes containing the diagram.
    """

    def __init__(self,gam,E0,rho,R=2.5,dr=.5,antiCorrelation=True,marker='o',
        s=40,ax=False,*opts,**keys):

        """Initialises the class given the pre-calculated metrics and draws
        the diagram grid with the first point.
        Markers in the diagram are colour-coded using the mean bias.

        Args:
            gam (float): STD ratio (Data/Reference)
            E0 (float): Normalised mean bias
            rho (float): Correlation Coefficient
            R (float): length of rays from origin in Taylor plot
            dr (float): step-size of circles around ideal point (1,0) in Taylor
                plot
            antiCorrelation (boolean): if True, show negative correlations
            marker: shape used to show points, should be a hollow shape as it is
                filled with colour code for bias.
            s (integer scalar or array_like): marker size in points
            ax (matplotlib.axes.Axes): axes to use, if False creates new Axes
            opts: positional arguments passed to ``add``
                function
            keys: keyword arguments passed to ``add`` function
        """

        Stats.__init__(self,gam,E0,rho)
        R=max(int(2.*self.gamma+1.)/2.,1.5)
        if ax:
            self.ax=ax
        else:
            self.ax=figure().add_subplot(111)
        f=self.ax.get_figure()
        self.drawTaylorGrid(R,dr,antiCorrelation)
        self._cmax=max(1.,abs(self.E0))
        self._cmin=-self._cmax
        self._lpos=[]
        if antiCorrelation:
            self._axis={'xmin':-1.3*R,'xmax':1.3*R,'ymin':-.1*R,'ymax':1.1*R}
        else:
            self._axis={'xmin':-.1*R,'xmax':1.3*R,'ymin':-.1*R,'ymax':1.1*R}
        mpl=self.add(self.gamma,self.E0,self.R,marker=marker,s=s,*opts,**keys)
        self.cbar=f.colorbar(mpl,ax=self.ax)
        self.cbar.set_label('${Bias}/\sigma_{ref}$')

    def __call__(self,gam,E0,rho,marker='o',s=40,*opts,**keys):

        """Adds points to the diagram adjusting the colour codes.

        Args:
            gam (float): STD ratio (Data/Reference)
            E0 (float): Normalised mean bias
            rho (float): Correlation Coefficient
            marker: shape used to show points, should be a hollow shape as it is
                filled with colour code for bias.
            s (integer scalar or array_like): marker size in points
            opts: positional arguments passed to ``add``
                function
            keys: keyword arguments passed to ``add`` function
        """
        Stats.__call__(self,gam,E0,rho,*opts,**keys)
        self._cmax=max(abs(self.E0),self._cmax)
        self._cmin=-self._cmax
        mpl=self.add(self.gamma,self.E0,self.R,marker=marker,s=s,*opts,**keys)
        self.cbar.set_clim(self._cmin,self._cmax)

    def add(self,gam,E0,R,marker='o',s=40,*opts,**keys):

        """Function to add additional points to the diagram, usually invoked
        by means of the ``__call__`` function.

        Args:
            gam (float): STD ratio (Data/Reference)
            E0 (float): Normalised mean bias
            R (float): Correlation Coefficient
            marker: shape used to show points, should be a hollow shape as it is
                filled with colour code for bias.
            s (integer scalar or array_like): marker size in points
            opts: positional arguments passed to matplotlib.pyplot.scatter
                function
            keys: keyword arguments passed to ``matplotlib.pyplot.scatter`` function
        Returns:
            matplotlib.collections.PathCollection instance from scatter call
        """

        E=rmsds(gam,R)
        self._lpos.append((gam*R,gam*sin(arccos(R)),E0))
        #replot previous values with fixed colourscale:
        for p in self._lpos:
            mpl=self.ax.scatter(atleast_1d(p[0]),atleast_1d(p[1]),c=atleast_1d(p[2]),
                vmin=self._cmin,vmax=self._cmax,marker=marker,s=s,*opts,**keys)
        self.ax.axis(**self._axis)
        return mpl

    def labels(self,lstr,*opts,**keys):

        """Adds labels ``lstr``` to the points in the diagram.

        Args:
            lstr (string): string for lable
            *opts: positional arguments passed to ``matplotlib.pyplot.text``
                function.
            **keys: keyword arguments passed to ``matplotlib.pyplot.text``
                function.
        """

        yrange=self.ax.axis()[2:]
        rmax=max(abs(yrange[1]-yrange[0]),1.5)
        for n,p in enumerate(self._lpos):
           self.ax.text(p[0]+.025*rmax,p[1]+.025*rmax,lstr[n],*opts,**keys)

class TargetDiagram(Target,Stats):

    """Class for drawing a Target diagram from pre-calculated metrics.

    Attributes:
          std (float): the standard deviation of the reference data
          E0 (float): normalised mean bias of the two data sets
          gama(float): the ratio of the std of data over reference data
          R(float): Pearson correlation
          p(float): p-value of Pearson correlation
          E(float): normalised root-mean square difference of the two dataset
          csv(string): collects the summary statistics of the instance to be
            written to csv file, when desired.
          ax (matplotlib.axes.Axes): axes containing the diagram.
    """

    def __init__(self,gam,E0,rho,marker='o',s=40,antiCorrelation=False,ax=False,
        *opts,**keys):

        """Initialises the class given the pre-calculated metrics and draws
        the diagram grid with the first point.
        Markers in the diagram are colour-coded using the mean bias.

        Args:
            gam (float): STD ratio (Data/Reference)
            E0 (float): normalised mean bias
            rho (float): Correlation Coefficient
            marker: shape used to show points, should be a hollow shape as it is
                filled with colour code for bias.
            s (integer scalar or array_like): marker size in points
            antiCorrelation (boolean): if True, show negative correlations
            ax (matplotlib.axes.Axes): axes to use, if False creates new Axes
            opts: positional arguments passed to ``add`` function
            keys: keyword arguments passed to ``add`` function
        """

        Stats.__init__(self,gam,E0,rho)
        if ax:
            self.ax=ax
        else:
            self.ax=figure().add_subplot(111)
        f=self.ax.get_figure()
        self.drawTargetGrid()
        if antiCorrelation:
          self._cmin=-1.
        else:
          self._cmin=0.
        self._cmax=1.
        self._lpos=[]
        mpl=self.add(self.gamma,self.E0,self.R,marker=marker,s=s,*opts,**keys)
        self.cbar=f.colorbar(mpl,ax=self.ax)
        self.cbar.set_label('Correlation Coefficient')

    def __call__(self,gam,E0,rho,marker='o',s=40,*opts,**keys):

        """Adds points to the diagram adjusting the colour codes

        Args:
            gam (float): STD ratio (Data/Reference)
            E0 (float): normalised mean bias
            rho (float): Correlation Coefficient
            marker: shape used to show points, should be a hollow shape as it is
                filled with colour code for bias.
            s (integer scalar or array_like): marker size in points
            opts: positional arguments passed to ``add`` function
            keys: keyword arguments passed to ``add`` function
        """

        Stats.__call__(self,gam,E0,rho)
        mpl=self.add(self.gamma,self.E0,self.R,marker=marker,s=s,*opts,**keys)

    def add(self,gam,E0,R,marker='o',s=40,*opts,**keys):

        """Function to add additional points to the diagram, usually invoked
        by means of the ``__call__`` function.

        Args:
            gam (float): STD ratio (Data/Reference)
            E0 (float): normalised mean bias
            R (float): Correlation Coefficient
            marker: shape used to show points, should be a hollow shape as it is
                filled with colour code for bias.
            opts: positional arguments passed to matplotlib.pyplot.scatter
                function
            keys: keyword arguments passed to ``matplotlib.pyplot.scatter`` function

        Returns:
            matplotlib.collections.PathCollection instance from scatter call
        """

        sig= gam>1 and 1 or -1
        E=sqrt(1.+gam**2-2.*gam*R)
        mpl=self.ax.scatter(atleast_1d(sig*E),atleast_1d(E0),c=atleast_1d(R),
            vmin=self._cmin,vmax=self._cmax,marker=marker,s=s,*opts,**keys)
        self._lpos.append((sig*E,E0,R))
        rmax=max(abs(array(self.ax.axis('scaled'))).max(),1.5)
        self.ax.plot((0,0),(-rmax,rmax),'k-')
        self.ax.plot((rmax,-rmax),(0,0),'k-')
        self.ax.axis(xmin=-rmax,xmax=rmax,ymax=rmax,ymin=-rmax)
        return mpl

    def labels(self,lstr,*opts,**keys):

        """Adds labels ``lstr``` to the points in the diagram.

        Args:
            lstr (string): string for lable
            *opts: positional arguments passed to ``matplotlib.pyplot.text``
                function.
            **keys: keyword arguments passed to ``matplotlib.pyplot.text``
                function.
        """

        rmax=max(abs(array(self.ax.axis())).max(),1.5)
        for n,p in enumerate(self._lpos):
           self.ax.text(p[0]+.025*rmax,p[1]+.025*rmax,lstr[n],*opts,**keys)

class TargetStatistics(StatsDiagram,TargetDiagram):

    """Class for drawing a Taylor diagram from input and reference data.

    Attributes:
          std (float): the standard deviation of the reference data
          E0 (float): normalised mean bias of the two data sets
          gama(float): the ratio of the std of data over reference data
          R(float): Pearson correlation
          p(float): the Pearson correlation with p-value
          E(float): mormalised root-mean square difference of the two dataset
          csv(string): collects the summary statistics of the instance to be
            written to csv file, when desired.
          ax (matplotlib.axes.Axes): axes containing the diagram.
    """

    def __init__(self,data,refdata,marker='o',s=40,antiCorrelation=False,
        ax=False,*opts,**keys):
        """Initialises the class computing all necessary metrics and draws
        the diagram grid with the first point.
        Markers in the diagram are colour-coded using the mean bias.

        Args:
            data(float array): input data
            refdata(float array): references data, same shape as input data
            marker: shape used to show points, should be a hollow shape as it is
                filled with colour code for bias.
            s (integer scalar or array_like): marker size in points
            antiCorrelation (boolean): if True, show negative correlations
            ax (matplotlib.axes.Axes): axes to use, if False creates new Axes
            opts: positional arguments passed to ``add`` function
            keys: keyword arguments passed to ``add`` function
        """
        StatsDiagram.__init__(self,data,refdata,*opts,**keys)
        if ax:
            self.ax=ax
        else:
            self.ax=figure().add_subplot(111)
        f=self.ax.get_figure()
        self.drawTargetGrid()
        if antiCorrelation:
          self._cmin=-1.
        else:
          self._cmin=0.
        self._cmax=1.
        self._lpos=[]
        mpl=self.add(self.gamma,self.E0,self.R,marker=marker,s=s,*opts,**keys)
        self.cbar=f.colorbar(mpl,ax=self.ax)
        self.cbar.set_label('Correlation Coefficient')

    def __call__(self,data,refdata,marker='o',s=40,*opts,**keys):

        """Adds points to the diagram adjusting the colour codes.

        Args:
            data(float array): input data
            refdata(float array): references data, same shape as input data
            marker: shape used to show points, should be a hollow shape as it is
                filled with colour code for bias.
            s (integer scalar or array_like): marker size in points
            opts: positional arguments passed to ``add`` function
            keys: keyword arguments passed to ``add`` function
        """
        StatsDiagram.__call__(self,data,refdata,*opts,**keys)
        mpl=self.add(self.gamma,self.E0,self.R,marker=marker,s=s,*opts,**keys)

class TaylorStatistics(StatsDiagram,TaylorDiagram):

    """Class for drawing a Taylor diagram from input and reference data.

    Attributes:
          std (float): the standard deviation of the reference data
          E0 (float): normalised mean bias of the two data sets
          gama(float): the ratio of the std of data over reference data
          R(float): Pearson correlation
          p(float): the Pearson correlation with p-value
          E(float):  normalised root-mean square difference of the two dataset
          csv(string): collects the summary statistics of the instance to be
            written to csv file, when desired.
          ax (matplotlib.axes.Axes): axes containing the diagram.
    """

    def __init__(self,data,refdata,R=2.5,dr=.5,antiCorrelation=True,marker='o',
        s=40,ax=False,*opts,**keys):
        """Initialises the class computing all necessary metrics and draws
        the diagram grid with the first point.
        Markers in the diagram are colour-coded using the mean bias.

        Args:
            data(float array): input data
            refdata(float array): references data, same shape as input data
            R (float): length of rays from origin in Taylor plot
            dr (float): step-size of circles around ideal point (1,0) in Taylor
                plot
            marker: shape used to show points, should be a hollow shape as it is
                filled with colour code for bias.
            s (integer scalar or array_like): marker size in points
            antiCorrelation (boolean): if True, show negative correlations
            ax (matplotlib.axes.Axes): axes to use, if False creates new Axes
            opts: positional arguments passed to ``add`` function
            keys: keyword arguments passed to ``add`` function
        """

        StatsDiagram.__init__(self,data,refdata,*opts,**keys)
        R=max(int(2.*self.gamma+1.)/2.,1.5)
        if ax:
            self.ax=ax
        else:
            self.ax=figure().add_subplot(111)
        f=self.ax.get_figure()
        self.drawTaylorGrid(R,dr,antiCorrelation)
        self._cmax=max(1.,abs(self.E0))
        self._cmin=-self._cmax
        self._lpos=[]
        if antiCorrelation:
            self._axis={'xmin':-1.3*R,'xmax':1.3*R,'ymin':-.1*R,'ymax':1.1*R}
        else:
            self._axis={'xmin':-.1*R,'xmax':1.3*R,'ymin':-.1*R,'ymax':1.1*R}
        mpl=self.add(self.gamma,self.E0,self.R,marker=marker,s=s,*opts,**keys)
        self.cbar=f.colorbar(mpl,ax=self.ax)
        self.cbar.set_label('${Bias}/\sigma_{ref}$')

    def __call__(self,data,refdata,marker='o',s=40,*opts,**keys):

        """Adds points to the diagram adjusting the colour codes.

        Args:
            data(float array): input data
            refdata(float array): references data, same shape as input data
            marker: shape used to show points, should be a hollow shape as it is
                filled with colour code for bias.
            s (integer scalar or array_like): marker size in points
            opts: positional arguments passed to ``add`` function
            keys: keyword arguments passed to ``add`` function
        """

        StatsDiagram.__call__(self,data,refdata,*opts,**keys)
        self._cmax=max(abs(self.E0),self._cmax)
        self._cmin=-self._cmax
        mpl=self.add(self.gamma,self.E0,self.R,marker=marker,s=s,*opts,**keys)
        self.cbar.set_clim(self._cmin,self._cmax)
