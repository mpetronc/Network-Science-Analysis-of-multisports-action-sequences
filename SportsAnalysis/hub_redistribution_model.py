"""Pre-period role-weighted hub replacement, preserving team passing mass."""
import numpy as np

def ec_index(A,ids):
 active=A.sum(0)+A.sum(1)>0;B=(A+A.T)[np.ix_(active,active)];_,v=np.linalg.eigh(B);shares=np.abs(v[:,-1]);shares/=shares.sum();hhi=float(shares@shares);n=len(shares);return dict(index=(hhi-1/n)/(1-1/n),hhi=hhi,n=n,shares=dict(zip(np.asarray(ids)[active].tolist(),shares.tolist())))

def redistribute(P,mp,early_abs_minutes,early_abs_games,hub,ids,positions):
 """Only training-period minutes and previous-season positions enter here."""
 mp=np.asarray(mp,float);assert mp[hub]>0 and P.sum()>0;base=240*mp/mp.sum();base_nohub=base.copy();base_nohub[hub]=0;eligible=mp>0;eligible[hub]=False;assert eligible.sum()*48>=240;capacity=np.maximum(0,48-base_nohub);hubpos=set(positions.get(ids[hub],[]));match=np.zeros(len(ids));rank=np.ones(len(ids));same=[]
 for i,pid in enumerate(ids):
  if not eligible[i]:continue
  pos=set(positions.get(pid,[]));match[i]=1. if hubpos and pos and hubpos&pos else (.5 if not hubpos or not pos else .25)
  if match[i]==1:same.append(i)
 for k,i in enumerate(sorted(same,key=lambda i:(-base[i],ids[i])),1):rank[i]=k
 fallback=capacity*match/rank;fallback[~eligible]=0;assert fallback.sum()>0;fallback/=fallback.sum();historical=np.zeros(len(ids))
 if early_abs_games:
  # present minutes are totals; present-game count supplied by caller through mp.
  historical=np.maximum(0,early_abs_minutes['per_game']-early_abs_minutes['present_per_game']);historical[~eligible]=0
 if historical.sum()>0:historical/=historical.sum();alpha=early_abs_games/(early_abs_games+5)
 else:alpha=0.
 weights=alpha*historical+(1-alpha)*fallback;weights[~eligible]=0;forecast=base_nohub.copy();remaining=240-forecast.sum()
 while remaining>1e-8:
  can=eligible&(forecast<48-1e-9);assert can.any();w=weights*can
  if w.sum()==0:w=capacity*can
  allocation=remaining*w/w.sum();extra=np.minimum(allocation,48-forecast);extra[~can]=0;assert extra.sum()>0;forecast+=extra;remaining=240-forecast.sum()
 gains=forecast-base_nohub;gains[~eligible]=0;replacement=gains/gains.sum();D=P.copy();D[hub]=0;D[:,hub]=0;M=D.copy();self_route_fallbacks=0
 def route(exclude):
  nonlocal self_route_fallbacks
  w=replacement.copy();w[exclude]=0;w[hub]=0
  if w.sum()==0:w=forecast.copy();w[exclude]=0;w[hub]=0;self_route_fallbacks+=1
  assert w.sum()>0;return w/w.sum()
 for j in range(len(ids)):
  if j!=hub and P[hub,j]>0:M[:,j]+=P[hub,j]*route(j)
 for i in range(len(ids)):
  if i!=hub and P[i,hub]>0:M[i,:]+=P[i,hub]*route(i)
 assert np.isclose(forecast.sum(),240) and forecast.max()<=48+1e-7;assert np.isclose(M.sum(),P.sum());assert M[hub].sum()+M[:,hub].sum()==0 and not np.diag(M).any();return M,dict(base_minutes=base.tolist(),forecast_minutes=forecast.tolist(),extra_minutes=gains.tolist(),replacement_weights=replacement.tolist(),fallback_role_weights=fallback.tolist(),historical_gain_weights=historical.tolist(),historical_blend=alpha,position_match=match.tolist(),position_depth_rank=rank.tolist(),hub_position=sorted(hubpos),self_route_fallbacks=self_route_fallbacks)
