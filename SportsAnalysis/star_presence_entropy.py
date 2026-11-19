"""Compare observed full passing entropy with/without the hub, plus a deletion diagnostic."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def bits(A):
 A=np.asarray(A,float);row=A.sum(-1);prob=np.divide(A,row[...,None],out=np.zeros_like(A),where=row[...,None]>0);return -np.sum(A*np.log2(np.maximum(prob,1e-300)),axis=(-2,-1))/A.sum(axis=(-2,-1))
def active_n(A):return int(np.count_nonzero(A.sum(0)+A.sum(1)))

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--star-presence-entropy',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path);p.add_argument('--draws',type=int,default=500);a=p.parse_args(argv);root=a.output_dir;casefolder=root/'star_dependency_study/network_case_studies';out=a.destination or casefolder/'star_presence_entropy';out.mkdir(parents=True,exist_ok=True);cases=pd.read_csv(casefolder/'nine_network_case_metrics.csv');rng=np.random.default_rng(20261120);rows=[];manifest=[];samples=[]
 for z in cases.itertuples():
  base=root/'2025-26/team_season_topics';ff=base/'networks'/z.team/'passing_matrix.csv';pf=base/'hub_absence_resilience/present_networks'/z.team/'passing_matrix.csv';F=pd.read_csv(ff,index_col=0);P=pd.read_csv(pf,index_col=0);assert F.index.astype(int).tolist()==P.index.astype(int).tolist() and F.columns.tolist()==P.columns.tolist();ids=F.index.astype(int).tolist();present=P.to_numpy(float);absent=F.to_numpy(float)-present;assert absent.min()>=-1e-8;absent=np.maximum(absent,0);i=ids.index(int(z.star_id));assert absent[i].sum()+absent[:,i].sum()==0
  assert np.isfinite(present).all() and (present>=0).all() and not np.diag(present).any() and not np.diag(absent).any()
  deleted=present.copy();deleted[i,:]=0;deleted[:,i]=0
  union=active_n(present+absent);den=np.log2(union-1);hp,hdel,ha=map(float,[bits(present),bits(deleted),bits(absent)]);N=int(min(present.sum(),absent.sum()));assert N>0
  # Equal TOTAL number of passes, retaining the star's row in present games.
  totals=[]
  for matrix in [present,absent]:
   counts=matrix.astype(np.int64);assert np.array_equal(matrix,counts)
   draw=rng.multivariate_hypergeometric(counts.ravel(),N,size=a.draws).reshape(a.draws,*counts.shape);assert (draw.sum(axis=(1,2))==N).all() and not np.diagonal(draw,axis1=1,axis2=2).any();totals.append(bits(draw)/den)
  delta=totals[1]-totals[0]
  rows.append(dict(team=z.team,season=z.season,star_id=z.star_id,star_name=z.star_name,offense_change=z.offense_change,core_case=z.core_case,union_active_players=union,present_active_players=active_n(present),absent_active_players=active_n(absent),present_passes=present.sum(),absent_passes=absent.sum(),star_outgoing_pass_share=present[i].sum()/present.sum(),passes_involving_star=present[i].sum()+present[:,i].sum(),entropy_with_star_bits=hp,entropy_present_star_deleted_bits=hdel,entropy_observed_without_star_bits=ha,total_change_bits=ha-hp,deletion_difference_bits=hdel-hp,additional_context_difference_bits=ha-hdel,entropy_with_star_common_norm=hp/den,entropy_present_star_deleted_common_norm=hdel/den,entropy_without_star_common_norm=ha/den,total_change_common_norm=(ha-hp)/den,deletion_difference_common_norm=(hdel-hp)/den,additional_context_difference_common_norm=(ha-hdel)/den,with_star_own_active_norm=hp/np.log2(active_n(present)-1),without_star_own_active_norm=ha/np.log2(active_n(absent)-1),equal_total_passes=N,equal_total_with_star_entropy_mean=totals[0].mean(),equal_total_without_star_entropy_mean=totals[1].mean(),equal_total_entropy_change_mean=delta.mean(),equal_total_entropy_change_sampling_sd=delta.std(ddof=1),equal_total_entropy_change_sampling_q025=np.quantile(delta,.025),equal_total_entropy_change_sampling_q975=np.quantile(delta,.975),previous_common_nonhub_count_matched_change=z.entropy_change))
  assert np.isclose((ha-hp),(hdel-hp)+(ha-hdel))
  samples.append(dict(team=z.team,with_star=totals[0].tolist(),without_star=totals[1].tolist()))
  for f in [ff,pf]:manifest.append(dict(team=z.team,path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 d=pd.DataFrame(rows);d.to_csv(out/'star_present_vs_absent_entropy.csv',index=False);pd.DataFrame(manifest).to_csv(out/'source_manifest.csv',index=False);(out/'equal_total_sampling_draws.json').write_text(json.dumps(samples));(out/'settings.json').write_text(json.dumps(dict(seed=20261120,draws=a.draws,normalization='Common log2(N_union_active-1) within each team; raw bits primary for decomposition',comparison='Observed full star-present vs observed full star-absent networks; synthetic deletion diagnostic separate',sampling='Equal TOTAL pass counts, not equal outgoing counts for each player, because hub has zero passes in absent games'),indent=2))
 with PdfPages(out/'star_present_vs_absent_entropy.pdf') as pdf:
  for subset,title,file in [(d[d.core_case],'Three central cases','core_three'),(d,'All nine available cases','all_nine')]:
   fig,axes=plt.subplots(1,len(subset),figsize=(max(15,3.1*len(subset)),6),squeeze=False);rangevalues=subset[['entropy_with_star_common_norm','entropy_present_star_deleted_common_norm','entropy_without_star_common_norm']].to_numpy();lo=rangevalues.min();hi=rangevalues.max();pad=max((hi-lo)*.12,.01)
   for ax,z in zip(axes.flat,subset.itertuples()):
    values=[z.entropy_with_star_common_norm,z.entropy_present_star_deleted_common_norm,z.entropy_without_star_common_norm];ax.plot(range(3),values,marker='o',c='#357b9c');ax.set_xticks(range(3),['Present\nfull network','Present\nstar deleted','Actually\nabsent']);ax.set_title(f'{z.team}: {z.star_name}\nTotal ΔH={z.total_change_common_norm:+.4f}',fontsize=10);ax.grid(alpha=.15)
    for j,v in enumerate(values):ax.annotate(f'{v:.4f}',(j,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    ax.set_ylabel('Outgoing conditional entropy, fixed-reference normalization',fontsize=8);ax.set_ylim(lo-pad,hi+pad);ax.margins(x=.15)
   fig.suptitle(title+': include the hub in the present-game entropy',fontsize=16);fig.text(.5,.015,'Blue values use all observed passing players; fixed within-team denominator log2(N_union−1). Middle state is a deletion diagnostic, not a simulated absence.\nActual absent-minus-present includes lost destinations, player composition, volume weights and behavioral/context differences. Passing edges only; season contexts, not episode windows.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.09,1,.92]);fig.savefig(out/(file+'.png'),dpi=150);pdf.savefig(fig);plt.close(fig)
  fig,ax=plt.subplots(figsize=(12,6));x=np.arange(len(d));ax.bar(x-.2,d.total_change_common_norm,width=.4,label='Observed full networks');ax.bar(x+.2,d.equal_total_entropy_change_mean,width=.4,label='Equal-total-pass sensitivity');ax.axhline(0,c='grey');ax.set_xticks(x,d.team);ax.set_ylabel('Entropy change: actually absent minus present WITH hub');ax.set_title('Does the total entropy change persist with equal pass-sample sizes?');ax.legend();fig.text(.5,.015,'Equal-total sampling preserves each context’s player mix; individual donors are not matched. Sampling ranges are not behavioral or population confidence intervals.\nThe earlier matched-nonhub Network 2→3 comparison answers a different question and is retained in the data.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.09,1,1]);fig.savefig(out/'total_change_comparison.png',dpi=150);pdf.savefig(fig);plt.close(fig)
 text='''# Entropy with the hub versus actual hub absence

All nine compatible 2025–26 passing-network cases. Entropy is outgoing H(receiver|passer), weighted by observed outgoing pass totals. Shooting edges are excluded.

Three NEW states are calculated: (1) all observed star-present passing, INCLUDING the hub; (2) the SAME present network with the hub's row and column zeroed, with no donor matching; (3) all observed star-absent passing. State (2) is a deletion diagnostic, not the earlier displayed count-matched Network 2. State (3) is the full observed absence network, not its earlier matched common-panel version. Earlier Network 2→3 summaries are preserved separately.

Primary decomposition in bits: full absence minus full presence = (synthetic deletion minus presence) + (full absence minus synthetic deletion). This is an algebraic decomposition, not a causal attribution. The second difference includes volume weights, players available, teammate co-appearance and context as well as possible reorganization. The first difference changes both the hub's outgoing contribution and everyone else's recipient probabilities/weights.

For readable normalized comparisons, use ONE denominator per team: log2(N−1), where N is the number of active players in the union of presence/absence networks. Keeping this denominator fixed avoids changing the scale when the star disappears. Maximum attainable entropy can be lower without the hub; that is part of the total comparison. Also export raw bits and own-active-count normalized values, but do not substitute differing denominators into the additive decomposition.

| Team | Hub | With star | Actually without star | Total change | Equal-total-pass change | Earlier matched-nonhub change |
|---|---|---:|---:|---:|---:|---:|
'''
 for z in d.itertuples():text+=f'| {z.team} | {z.star_name} | {z.entropy_with_star_common_norm:.4f} | {z.entropy_without_star_common_norm:.4f} | {z.total_change_common_norm:+.4f} | {z.equal_total_entropy_change_mean:+.4f} | {z.previous_common_nonhub_count_matched_change:+.4f} |\n'
 text+='''
Equal-total sample-size sensitivity: for both full networks sample without replacement to the smaller TOTAL pass count, 500 times. This retains the hub and their passes in the present state. Matching every donor to the absent state would force the hub's row to zero and defeat the purpose of this new comparison. Finite-sample effects are checked, but playing time, lineup opportunity and donor composition are not controlled. Sampling SD/quantiles describe subsampling variation, not uncertainty in a causal behavioral effect.

These total changes can differ in size or sign from the earlier comparison that removed the hub in both contexts and matched remaining donors. Higher entropy is variety, not offensive quality. The cases and full-season contexts are observational; exact lineup overlap and episode-specific passing are unavailable. Prior baseline-subtracted entropy regressions are different analyses.
'''
 (out/'RESULTS_AND_METHOD.md').write_text(text);print(d[['team','star_name','entropy_with_star_common_norm','entropy_without_star_common_norm','total_change_common_norm','equal_total_entropy_change_mean','previous_common_nonhub_count_matched_change']].to_string(index=False));return 0
if __name__=='__main__':raise SystemExit(main())
