"""Descriptive team-level regressions of entropy change on passing EC concentration."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def holm(p):
 p=np.asarray(p);order=np.argsort(p);adjusted=np.empty(len(p));adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))));return adjusted

class Fit:
 def __init__(self,x,y):
  X=np.column_stack([np.ones(len(x)),x]);self.inv=np.linalg.inv(X.T@X);self.params=self.inv@X.T@y;res=y-X@self.params;h=np.sum((X@self.inv)*X,axis=1);self.cov=self.inv@(X.T@((res/(1-h))[:,None]**2*X))@self.inv;self.df=len(x)-2;self.se=np.sqrt(np.diag(self.cov));self.pvalues=2*st.t.sf(np.abs(self.params/self.se),self.df);self.rsquared=1-(res@res)/np.sum((y-y.mean())**2)
 def conf_int(self):
  width=st.t.ppf(.975,self.df)*self.se;return np.column_stack([self.params-width,self.params+width])
 def prediction(self,x):
  X=np.column_stack([np.ones(len(x)),x]);mean=X@self.params;width=st.t.ppf(.975,self.df)*np.sqrt(np.maximum(0,np.sum((X@self.cov)*X,axis=1)));return pd.DataFrame({'mean':mean,'mean_ci_lower':mean-width,'mean_ci_upper':mean+width})
