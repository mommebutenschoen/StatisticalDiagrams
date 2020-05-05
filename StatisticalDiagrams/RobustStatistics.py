import logging
from numpy import median,empty,logical_or,arange,round
from numpy.ma import getmaskarray,getdata,masked_where
try:
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb
from scipy.stats.mstats import mquantiles
try:
    from itertools import izip as zip
except:
    pass

_nearest = lambda x:int(round(s))

def binomcoeff(n,k):

    """Binomial coefficient.

    Args:
        n (integer): positive integer number
        k (integer): positive integer number <= n

    Returns:
        Binomial coefficient (integer).
    """

    _nearest(comb(n,k))

def IQR(data):

    """Inter-quartile range of input data.

    Args:
        data (float array-like): input data

    Returns:
        Inter-quartile range (float).
    """

    q25,q75=mquantiles(data,prob=[.25,.75])
    return q75-q25

def flatCompress(data):

   """Flattened and compressed (removing masekd values) representation of input
   data.

   Args:
      data (array-like): input data

   Returns:
      flattened and compressed version of input data.
   """

   if any(getmaskarray(data)):
      return data.ravel().compressed()
   else:
      return getdata(data).ravel()

def flatCompressSamples(s1,s2):

   """Flattened and compressed representation of two datasets using the combined
   mask of the two (by disjunction).

   Args:
      s1 (array-like): first input dataset
      s2 (array-like): second input dataset, same shape as first

   Returns:
      Flattened and compressed representation of input datasets.
   """

   Mask=logical_or(getmaskarray(s1),getmaskarray(s2))
   s1=masked_where(Mask,s1).ravel().compressed()
   s2=masked_where(Mask,s2).ravel().compressed()
   return s1,s2

def MIDExplicit(data):

    """Median Interpoint Difference::

        MID=med|xi-xj|,i<j

    Args:
        data (float or integer array-like): input data

    Returns:
        Median interpoint distance.
    """

    d=[]
    for n,x in enumerate(data.ravel()):
        for y in data.ravel()[n:]:
           d.append(abs(x-y))
    d.sort()
    k=(n+1)*n//2//2
    logging.info("Moment",k,"of",n+1)
    return d[k]

def Sn(data):

    """Robust Scale measure::

       Sn = 1.1926 lowmed(i=1,n)( highmed(j=1,n)(|xi - xj|) )

    Optimised Version
    See:
    Rousseeuw, P. J. & Croux
    C. Alternatives to the Median Absolute Deviation
    Journal of the American Statistical Association
    1993, 88, 1273-1283
    Croux, C. & Rousseeuw, P. J.
    Time-efficient algorithms for two highly robust estimators of scale
    Computational Statistics
    1992, 1

    Args:
        data (float array-like): input data

    Returns:
        Sn
    """

    y=data.flatten()
    if y.shape[0]<2:
        if y.shape[0]==1:
            return 0.
        else:
            return None
    n=y.shape[0]
    nlm=(n+1)//2 #low median index
    y.sort()
    a2=empty(y.shape)
    a2[0]=y[n//2]-y[0]
    for i,yy in zip(arange(1,nlm),y[1:nlm]):
        nA=i
        nB=n-i-1
        diff=nB-nA
        leftA=1
        leftB=1
        rightA=nB
        rightB=nB
        Amin=diff//2+1
        Amax=diff//2+nA
        while leftA<rightA:
            length=rightA-leftA+1
            even=1-length%2
            half=(length-1)//2
            tryA=leftA+half
            tryB=leftB+half
            if tryA<Amin:
                rightB=tryB
                leftA=tryA+even
            elif tryA>Amax:
                rightA=tryA
                leftB=tryB+even
            else:
                medA=yy-y[i-1-tryA+Amin]
                medB=y[tryB+i]-yy
                if medA>=medB:
                    rightA=tryA
                    leftB=tryB+even
                else:
                    rightB=tryB
                    leftA=tryA+even
        if leftA>Amax:
            a2[i]=y[leftB+i]-yy
        else:
            medA=yy-y[i-1-leftA+Amin]
            medB=y[leftB+i]-yy
            a2[i]=min(medA,medB)
    for i,yy in zip(arange(nlm,n-1),y[nlm:n-1]):
        nA=n-i-1
        nB=i
        diff=nB-nA
        leftA=1
        leftB=1
        rightA=nB
        rightB=nB
        Amin=diff//2+1
        Amax=diff//2+nA
        while leftA<rightA:
            length=rightA-leftA+1
            even=1-length%2
            half=(length-1)//2
            tryA=leftA+half
            tryB=leftB+half
            if tryA<Amin:
                rightB=tryB
                leftA=tryA+even
            elif tryA>Amax:
                rightA=tryA
                leftB=tryB+even
            else:
                medA=y[i+tryA-Amin+1]-yy
                medB=yy-y[i-tryB]
                if medA>=medB:
                    rightA=tryA
                    leftB=tryB+even
                else:
                    rightB=tryB
                    leftA=tryA+even
        if leftA>Amax:
            a2[i]=yy-y[i-leftB]
        else:
            medA=y[i+1+leftA-Amin]-yy
            medB=yy-y[i-leftB]
            a2[i]=min(medA,medB)
    a2[n-1]=y[n-1]-y[nlm-1]
    if n<10:
        if n==2: cn=.743
        elif n==3: cn=1.851
        elif n==4: cn=.954
        elif n==5: cn=1.351
        elif n==6: cn=.993
        elif n==7: cn=1.198
        elif n==8: cn=1.005
        elif n==9: cn=1.131
    elif n%2==1:cn=n/(n-.9)
    else: cn=1.
    return cn*1.1926*_orderStatistic(a2,nlm,n)

def SnExplicit(data,c=1.1926):

   """ Robust Scale measure::

      Sn = 1.1926 med(i=1,n)( med(j=1,n)(|xi - xj|) )

   In this version the mathematical definition was directly translated
   into code. This is highly inefficient (time and memory) and should only
   be used for checks.
   See:
   Rousseeuw, P. J. & Croux
   C. Alternatives to the Median Absolute Deviation
   Journal of the American Statistical Association
   1993, 88, 1273-1283
   Croux, C. & Rousseeuw, P. J.
   Time-efficient algorithms for two highly robust estimators of scale
   Computational Statistics
   1992, 1
   Args:
        data (float array-like): input data

   Returns:
        Sn
   """
   data=flatCompress(data)
   if data.shape[0]<2:
        if data.shape[0]==1:
            return 0.
        else:
            return None
   dists=empty(data.shape)
   dd=empty(data.shape)
   nsize=data.shape[0]
   for n,d in enumerate(data):
      dists[n]=_orderStatistic(abs(d-data),nsize//2+1,nsize)
   if nsize<10:
        if nsize==2: cn=.743
        elif nsize==3: cn=1.851
        elif nsize==4: cn=.954
        elif nsize==5: cn=1.351
        elif nsize==6: cn=.993
        elif nsize==7: cn=1.198
        elif nsize==8: cn=1.005
        elif nsize==9: cn=1.131
   else:
        if nsize%2==1:cn=nsize/(nsize-.9)
        else: cn=1.
   return cn*c*_orderStatistic(dists,(nsize+1)//2,nsize)

def Sdist(s1,s2,distFun):

   """Computes distance scale between to sets of samples based on Sn metric.

   Args:
      s1, s2 (float array-likes): input datasets, same shape.
      distFun: distance function to be used

   Returns:
      Median distance based on
   """

   dists1=empty(s1.shape)
   for n,d1 in enumerate(s1):
       dists2=empty(s2.shape)
       for m,d2 in  enumerate(s2):
           dists2[m]=distFun(d1,d2)
       dists1[n]=median(dists2)
   return median(dists1)

def MID(data):

    """ Median interpoint distance, merory efficient version
    adopted from:
    Rousseeuw, P. J. & Croux
    C. Alternatives to the Median Absolute Deviation
    Journal of the American Statistical Association
    1993, 88, 1273-1283
    Croux, C. & Rousseeuw, P. J.
    Time-efficient algorithms for two highly robust estimators of scale
    Computational Statistics
    1992, 1

    Args:
       data (float array-like): input data

    Returns:
        Median interpoint distance (float).
    """

    y=data.flatten()
    if y.shape[0]<2:
        if y.shape[0]==1:
            return 0.
        else:
            return None
    n=y.shape[0]
    k=n*(n-1)//2//2
    logging.info("Moment",k,"of",n)
    y.sort()
    left=n+1 -arange(n)
    right=arange(n)+1
    jhelp=n*(n+1)//2
    knew=k+jhelp
    nL=jhelp
    nR=n*n
    found=False
    P=empty(n,dtype=int)
    Q=empty(n,dtype=int)
    weight=empty(n,dtype=int)
    work=empty(n)
    while (nR-nL>n) and not found:
        j=0
        for i,l,r,yy in zip(arange(1,n),left[1:],right[1:],y[1:]):
            if l<=r:
                w=r-l+1
                jhelp=l+w//2
                weight[j]=w
                work[j]=yy-y[n-jhelp]
                j+=1
        trial=_whimed(work[:j],weight[:j])
        j=0
        for i,yy in zip(arange(n-1,-1,-1),y[::-1]):
            while j<n and yy-y[n-j-1]<trial:
                j+=1
            P[i]=j
        j=n+1
        for i,yy in zip(arange(n),y):
            while yy-y[n-j+1]>trial:
                j-=1
            Q[i]=j
        sumP=P.sum()
        sumQ=(Q-1).sum()
        if knew<=sumP:
            right=P.copy()
            nR=sumP
        else:
            if knew>sumQ:
                left=Q.copy()
                nL=sumQ
            else:
                qn=trial
                found=True
    if not found:
        j=0
        for i,l,r,yy in zip(arange(1,n),left[1:],right[1:],y[1:]):
            if l<=r:
                for jj in arange(l,r+1):
                    work[j]=yy-y[n-jj]
                    j+=1
        qn=_orderStatistic(work,knew-nL,j)
    if n<10:
        if n==2: dn=.399
        elif n==3: dn=.994
        elif n==4: dn=.512
        elif n==5: dn=.844
        elif n==6: dn=.611
        elif n==7: dn=.857
        elif n==8: dn=.669
        elif n==9: dn=.872
    else:
        if n%2==1: dn=n/(n+1.4)
        else: dn=n/(n+3.8)
    return dn*qn

def QnExplicit(data,c=2.2219):

   """Robust scale measure::

      Qn = c*dn*{|xi-xj|;i<j}_(k)

   the kth order statistic of the ``( n over 2 )`` interpoint distances
   ``k=(n/2+1 over 2)``.
   In this version the mathematical definition was directly translated
   into code. This is highly inefficient (time and memory) and should only
   be used for checks.

   See:

      Rousseeuw, P. J. & Croux
      C. Alternatives to the Median Absolute Deviation
      Journal of the American Statistical Association
      1993, 88, 1273-1283
      Croux, C. & Rousseeuw, P. J.
      Time-efficient algorithms for two highly robust estimators of scale
      Computational Statistics
      1992, 1

   Args:
      data (float array-like): input data
      c (float): scaling multiplyer (see background paper)
      
   Returns:
      Qn (float).
   """

   data=flatCompress(data)
   if data.shape[0]<2:
        if data.shape[0]==1:
            return 0.
        else:
            return None
   n=data.shape[0]
   if n<10:
        if n==2: dn=.399
        elif n==3: dn=.994
        elif n==4: dn=.512
        elif n==5: dn=.844
        elif n==6: dn=.611
        elif n==7: dn=.857
        elif n==8: dn=.669
        elif n==9: dn=.872
   else:
        if n%2==1: dn=n/(n+1.4)
        else: dn=n/(n+3.8)
   dists=empty(_nearest(binomcoeff(n,2)))
   k=0
   for j,d1 in enumerate(data):
      for m,d2 in enumerate(data[:j]):
         dists[k]=abs(d1-d2)
         k+=1
   kord=binomcoeff(n/2+1,2)
   return c*dn*_orderStatistic(dists,kord,k)

def Qn(data):

    """Robust scale measure::

       Qn = c*dn*{|xi-xj|;i<j}_(k)

    the kth order statistic of the ``( n over 2 )`` interpoint distances
    ``k=(n/2+1 over 2)``.
    Optimised version.

    See:

       Rousseeuw, P. J. & Croux
       C. Alternatives to the Median Absolute Deviation
       Journal of the American Statistical Association
       1993, 88, 1273-1283
       Croux, C. & Rousseeuw, P. J.
       Time-efficient algorithms for two highly robust estimators of scale
       Computational Statistics
       1992, 1

    Args:
       data (float array-like): input data
       c (float): scaling multiplyer (see background paper)

    Returns:
       Qn (float).
    """

    y=data.flatten()
    if y.shape[0]<2:
        if y.shape[0]==1:
            return 0.
        else:
            return None
    n=y.shape[0]
    h=n//2+1
    k=h*(h-1)//2
    y.sort()
    left=n+1 -arange(n)
    right=arange(n)+1
    jhelp=n*(n+1)//2
    knew=k+jhelp
    nL=jhelp
    nR=n*n
    found=False
    P=empty(n,dtype=int)
    Q=empty(n,dtype=int)
    weight=empty(n,dtype=int)
    work=empty(n)
    while (nR-nL>n) and not found:
        j=0
        for i,l,r,yy in zip(arange(1,n),left[1:],right[1:],y[1:]):
            if l<=r:
                w=r-l+1
                jhelp=l+w//2
                weight[j]=w
                work[j]=yy-y[n-jhelp]
                j+=1
        trial=_whimed(work[:j],weight[:j])
        j=0
        for i,yy in zip(arange(n-1,-1,-1),y[::-1]):
            while j<n and yy-y[n-j-1]<trial:
                j+=1
            P[i]=j
        j=n+1
        for i,yy in zip(arange(n),y):
            while yy-y[n-j+1]>trial:
                j-=1
            Q[i]=j
        sumP=P.sum()
        sumQ=(Q-1).sum()
        if knew<=sumP:
            right=P.copy()
            nR=sumP
        else:
            if knew>sumQ:
                left=Q.copy()
                nL=sumQ
            else:
                qn=trial
                found=True
    if not found:
        j=0
        for i,l,r,yy in zip(arange(1,n),left[1:],right[1:],y[1:]):
            if l<=r:
                for jj in arange(l,r+1):
                    work[j]=yy-y[n-jj]
                    j+=1
        qn=_orderStatistic(work,knew-nL,j)
    if n<10:
        if n==2: dn=.399
        elif n==3: dn=.994
        elif n==4: dn=.512
        elif n==5: dn=.844
        elif n==6: dn=.611
        elif n==7: dn=.857
        elif n==8: dn=.669
        elif n==9: dn=.872
    else:
        if n%2==1: dn=n/(n+1.4)
        else: dn=n/(n+3.8)
    return dn*2.2219*qn

def Qdist(s1,s2,distFun):

   """Computes distance scale between to sets of samples based on Sn scale.

     Args:
        s1, s2 (float array-likes): input datasets, same shape.
        distFun: distance function to be used

     Returns:
        Median distance based on
    """

   dists=empty(s1.shape[0]*s2,shape[0])
   k=0
   for d1 in s1:
       for d2 in s2:
           dists[k]=distFun(d1,d2)
           k+=1
   return mquantiles(dists,[.25,])[0]


def MAD(data,c=1.4826):

   """Computes Median Absolute Deviation::

         MAD=c*med|xi-med(xi)|

    Args:
        data (float array-like): input data
        c (float): scaling coefficient

    Returns:
        Median absolute deviation (float).

    """

   data=flatCompress(data)
   return c*median(abs(data-median(data)))

def MAE(x,y):

   """Computes Median Absolute Error::

         MAE=med|xi-yi||

   Args:
      xi,yi (float array-like): datasets to compare, same shape

   Returns:
      Median Absolute Error (float).
   """

   x,y=flatCompressSamples(x,y)
   return median(abs(x-y))

def MedianBias(x,y):

   """Computes Median Bias::

      MB=med|xi|-med|yi||

   Args:
      x,y (float array-like): datasets to compare, same shape

   Returns:
      Median bias (float).
   """
   x,y=flatCompressSamples(x,y)
   return median(x)-median(y)

def unbiasedMAE(x,y):

   """Computes unbiased Median Absolute Error::

       uMAE=med|xi-yi-med(xi)+med(yi)||

   Args:
      x,y (float array-like): datasets to compare, same shape

   Returns:
      Median bias (float).
   """

   x,y=flatCompressSamples(x,y)
   return median(abs(x-y-median(x)+median(y)))

def _orderStatistic(data,nq,n):
    x=data.ravel()[:n]
    x.sort()
    return x[nq-1]

def _whimed(a,iw):
    nn=a.shape[0]
    acand=empty(nn)
    iwcand=empty(nn)
    wtotal=iw.sum()
    wrest=0
    while True:
      trial=_orderStatistic(a,nn//2+1,nn)
      wleft=0
      wmid=0
      wright=0
      for i,aa,w in zip(arange(nn),a[:nn],iw[:nn]):
        if aa<trial:
            wleft+=w
        elif aa>trial:
            wright+=w
        else:
            wmid+=w
      if 2*wrest+2*wleft>wtotal:
        kcand=0
        for i,aa,w in zip(arange(nn),a[:nn],iw[:nn]):
            if aa<trial:
                acand[kcand]=aa
                iwcand[kcand]=w
                kcand+=1
        nn=kcand
      elif 2*wrest+2*wleft+2*wmid>wtotal:
        return trial
      else:
        kcand=0
        for i,aa,w in zip(arange(nn),a[:nn],iw[:nn]):
            if (aa>trial):
                acand[kcand]=aa
                iwcand[kcand]=w
                kcand+=1
        nn=kcand
        wrest+=wleft+wmid
      for i,aa,w in zip(arange(nn),acand[:nn],iwcand[:nn]):
        a[i]=aa
        iw[i]=w
