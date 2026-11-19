"""All-30 coverage, observed entropy with/without each season's passing hub, with missingness explicit."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from star_presence_entropy import bits,active_n

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--league-star-presence-entropy',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--data-dir',type=Path);p.add_argument('--destination',type=Path);p.add_argument('--draws',type=int,default=500);a=p.parse_args(argv)
 root=a.output_dir;data=a.data_dir or root/'league_star_presence_entropy';out=a.destination or data;out.mkdir(parents=True,exist_ok=True);coveragefile=data/'coverage_plan.csv';coverage=pd.read_csv(coveragefile);assert len(coverage)==30 and not coverage.team.duplicated().any();priorfile=root/'court_zone_networks/regressions/four_metrics.csv';prior=pd.read_csv(priorfile);prior=prior[prior.season.eq('2024-25')].set_index('team');playerfile=root/'2025-26/team_season_topics/player_games.csv';players=pd.read_csv(playerfile,dtype={'GAME_ID':str});rng=np.random.default_rng(20261121);rows=[];manifest=[];draws=[]
 def record(f,team='ALL'):manifest.append(dict(team=team,path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 for f in [coveragefile,priorfile,playerfile]:record(f)
 for z in coverage.itertuples():
  ff=root/'2025-26/team_season_topics/networks'/z.team/'passing_matrix.csv';F=pd.read_csv(ff,index_col=0);ids=F.index.astype(int).tolist();T=F.to_numpy(float);assert ids==F.columns.astype(int).tolist() and np.isfinite(T).all() and (T>=0).all() and not np.diag(T).any();star=ids.index(int(z.hub_id));un=active_n(T);den=np.log2(un-1);record(ff,z.team)
  # Independently confirm that the chosen hub is derived from passing-only EC.
  mask=T.sum(0)+T.sum(1)>0;V=T[np.ix_(mask,mask)];_,e=np.linalg.eigh(V+V.T);v=np.abs(e[:,-1]);v/=v.sum();ec=dict(zip(np.array(ids)[mask],v));assert np.isclose(ec[int(z.hub_id)],max(v),atol=1e-8)
  old=prior.loc[z.team];row=dict(team=z.team,season='2025-26',hub_id=int(z.hub_id),hub_name=z.hub_name,hub_definition='Full-season 2025–26 passing-only EC leader; retrospective descriptive role',prior_season_hub_id=int(old.passing_leader_id),prior_season_hub_name=old.passing_leader_name,same_as_prior_season_hub=int(z.hub_id)==int(old.passing_leader_id),present_games=z.present_games,absent_games=z.absent_games,few_absence_games=0<z.absent_games<10,union_active_players=un,full_season_passes=T.sum(),hub_other_NBA_teams=int(players[players.PLAYER_ID.eq(z.hub_id)].TEAM_ID.nunique())-1)
  fields=['present_passes','absent_passes','with_hub_bits','present_hub_deleted_bits','without_hub_bits','total_change_bits','deletion_difference_bits','additional_context_difference_bits','with_hub_entropy','present_hub_deleted_entropy','without_hub_entropy','entropy_change','equal_total_passes','equal_total_with_hub_entropy','equal_total_without_hub_entropy','equal_total_entropy_change','sampling_sd','sampling_q025','sampling_q975']
  row.update(hub_ec_share=float(ec[int(z.hub_id)]),passing_ec_hhi=float(v@v),passing_ec_concentration=float((v@v-1/len(v))/(1-1/len(v))));row.update({f:np.nan for f in fields});P=None;A=None;reason=''
  if z.absent_games==0:P=T.copy();A=np.zeros_like(T);status='no_observed_absence'
  elif z.validated_legacy_network:
   pf=root/'2025-26/team_season_topics/hub_absence_resilience/present_networks'/z.team/'passing_matrix.csv';Pframe=pd.read_csv(pf,index_col=0);assert Pframe.index.astype(int).tolist()==ids and Pframe.columns.tolist()==F.columns.tolist();P=Pframe.to_numpy(float);A=T-P;status='complete_cached';record(pf,z.team)
  else:
   folder=data/'new_networks'/z.team;auditfile=folder/'audit.json'
   if auditfile.exists():
    audit=json.loads(auditfile.read_text());record(auditfile,z.team)
    if audit['complete']:
     pf=folder/'present_passing_matrix.csv';af=folder/'absent_passing_matrix.csv';Pframe=pd.read_csv(pf,index_col=0);Aframe=pd.read_csv(af,index_col=0);assert Pframe.index.astype(int).tolist()==ids and Aframe.index.astype(int).tolist()==ids;P=Pframe.to_numpy(float);A=Aframe.to_numpy(float);status='complete_new';record(pf,z.team);record(af,z.team)
    else:status='unverified_profile_or_partition';reason='; '.join(audit['errors'])
   else:status='missing_downloaded_network';reason='No validated reconstructed network found'
  row['status']=status;row['reason']=reason
  if P is not None:
   assert A.min()>=-1e-8 and P.min()>=0 and np.allclose(P+A,T);A=np.maximum(A,0);assert A[star].sum()+A[:,star].sum()==0 and not np.diag(A).any()
   deleted=P.copy();deleted[star]=0;deleted[:,star]=0;hp=float(bits(P));hd=float(bits(deleted));row.update(present_passes=P.sum(),absent_passes=A.sum(),with_hub_bits=hp,present_hub_deleted_bits=hd,with_hub_entropy=hp/den,present_hub_deleted_entropy=hd/den)
   if z.absent_games>0:
    assert A.sum()>0;ha=float(bits(A));row.update(without_hub_bits=ha,total_change_bits=ha-hp,deletion_difference_bits=hd-hp,additional_context_difference_bits=ha-hd,without_hub_entropy=ha/den,entropy_change=(ha-hp)/den);N=int(min(P.sum(),A.sum()));pair=[]
    for M in [P,A]:
     C=M.astype(int);assert np.array_equal(M,C);S=rng.multivariate_hypergeometric(C.ravel(),N,size=a.draws).reshape(a.draws,*C.shape);assert np.all(S.sum((1,2))==N) and not np.diagonal(S,axis1=1,axis2=2).any();pair.append(bits(S)/den)
    delta=pair[1]-pair[0];row.update(equal_total_passes=N,equal_total_with_hub_entropy=pair[0].mean(),equal_total_without_hub_entropy=pair[1].mean(),equal_total_entropy_change=delta.mean(),sampling_sd=delta.std(ddof=1),sampling_q025=np.quantile(delta,.025),sampling_q975=np.quantile(delta,.975));draws.append(dict(team=z.team,with_hub=pair[0].tolist(),without_hub=pair[1].tolist()));assert np.isclose(row['total_change_bits'],row['deletion_difference_bits']+row['additional_context_difference_bits'])
  rows.append(row)
 d=pd.DataFrame(rows).sort_values(['passing_ec_concentration','team'],ascending=[False,True]).reset_index(drop=True);d.insert(0,'ec_rank',np.arange(1,len(d)+1));d.to_csv(out/'all_30_team_entropy.csv',index=False);d[['ec_rank','team','hub_name','hub_ec_share','passing_ec_concentration','entropy_change']].to_csv(out/'teams_ranked_by_ec_index.csv',index=False);valid=d[d.status.isin(['complete_cached','complete_new'])].copy();valid.to_csv(out/'available_team_comparisons.csv',index=False);d[d.same_as_prior_season_hub].to_csv(out/'unchanged_hub_definition_subset.csv',index=False);pd.DataFrame(manifest).to_csv(out/'source_manifest.csv',index=False);(out/'equal_total_sampling_draws.json').write_text(json.dumps(draws))
 oldfile=root/'star_dependency_study/network_case_studies/star_presence_entropy/star_present_vs_absent_entropy.csv'
 if oldfile.exists():
  old=pd.read_csv(oldfile);check=valid.merge(old,on='team',suffixes=('_new','_old'),validate='one_to_one');assert len(check)==9 and (check.hub_id==check.star_id).all()
  assert np.allclose(check.with_hub_entropy,check.entropy_with_star_common_norm) and np.allclose(check.without_hub_entropy,check.entropy_without_star_common_norm)
 with PdfPages(out/'all_teams_star_presence_entropy.pdf') as pdf:
  fig,ax=plt.subplots(figsize=(15,8));v=valid.sort_values('ec_rank');x=np.arange(len(v));ax.bar(x-.2,v.entropy_change,width=.4,label='Observed full networks');ax.bar(x+.2,v.equal_total_entropy_change,width=.4,label='Equal-total pass sensitivity');ax.axhline(0,c='grey');ax.set_xticks(x,[f'{z.ec_rank}. {z.team}' for z in v.itertuples()],rotation=45);ax.set_ylabel('Outgoing entropy: hub absent minus hub present');ax.set_title(f'2025–26 | {len(v)} observed comparisons from all 30 teams\nOrdered by EC concentration index, highest first; missing absence comparisons are not zeros');ax.legend();fig.text(.5,.015,'Within-team denominator fixed at log2(N_union−1). Full-season present/absent contexts; not lineup- or game-window matched.\nSmall absence samples are flagged in the data. Current-season hub selection is descriptive/retrospective; the prior-season-identity study remains separate.',ha='center',fontsize=10);fig.tight_layout(rect=[0,.08,1,.93]);fig.savefig(out/'league_entropy_changes.png',dpi=150);pdf.savefig(fig);plt.close(fig)
  for start in range(0,30,6):
   subset=d.iloc[start:start+6];fig,axes=plt.subplots(2,3,figsize=(17,10));values=subset[['with_hub_entropy','present_hub_deleted_entropy','without_hub_entropy']].to_numpy();lo=np.nanmin(values);hi=np.nanmax(values);pad=max((hi-lo)*.15,.02)
   for ax,z in zip(axes.flat,subset.itertuples()):
    series=[z.with_hub_entropy,z.present_hub_deleted_entropy,z.without_hub_entropy];ax.plot(range(3),series,marker='o',c='#377d9e');ax.set_xticks(range(3),['With hub','Deletion check','Actually absent']);ax.set_ylim(lo-pad,hi+pad);ax.set_xlim(-.25,2.25);ax.grid(alpha=.15)
    for j,value in enumerate(series):
     if np.isfinite(value):ax.annotate(f'{value:.4f}',(j,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
    if not np.isfinite(z.without_hub_entropy):ax.text(2,(lo+hi)/2,'No observed absence' if z.absent_games==0 else 'Unverified data',ha='center',fontsize=9,color='#9c463b',rotation=90)
    change=f'{z.entropy_change:+.4f}' if np.isfinite(z.entropy_change) else 'N/A';ax.set_title(f'#{z.ec_rank} {z.team}: {z.hub_name} | EC index {z.passing_ec_concentration:.4f}\nPresent {z.present_games}, absent {z.absent_games} games | ΔH={change}',fontsize=10);ax.set_ylabel('Fixed-reference normalized outgoing entropy',fontsize=8)
   fig.suptitle(f'EC concentration index, highest first | ranks {start+1}–{start+len(subset)}',fontsize=17);fig.text(.5,.015,'Deletion check removes hub row/column from the SAME present network; it is not an observed absence or the earlier matched-nonhub panel.\nRaw bits, a fixed within-team denominator and equal-total sample sensitivity are exported. Variety is not efficiency; no causal claim.',ha='center',fontsize=10);fig.tight_layout(rect=[0,.08,1,.94]);fig.savefig(out/f'teams_{start+1:02d}_{start+len(subset):02d}.png',dpi=150);pdf.savefig(fig);plt.close(fig)
  fig,ax=plt.subplots(figsize=(9,8));ax.scatter(valid.with_hub_entropy,valid.without_hub_entropy,s=35+4*valid.absent_games,c=np.where(valid.few_absence_games,'#ca8b3e','#357d9c'),alpha=.75)
  for z in valid.itertuples():ax.annotate(z.team,(z.with_hub_entropy,z.without_hub_entropy),fontsize=8,xytext=(3,3),textcoords='offset points')
  lo=min(valid.with_hub_entropy.min(),valid.without_hub_entropy.min())-.02;hi=max(valid.with_hub_entropy.max(),valid.without_hub_entropy.max())+.02;ax.plot([lo,hi],[lo,hi],c='grey',ls='--');ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_xlabel('Entropy with hub');ax.set_ylabel('Entropy actually without hub');ax.set_title('Above diagonal: more diverse passing without hub\nOrange: fewer than 10 absence games; size reflects absence games');fig.tight_layout();fig.savefig(out/'with_vs_without_all_teams.png',dpi=150);pdf.savefig(fig);plt.close(fig)
 from league_star_presence_report import write_report
 write_report(out)
 print(d[['team','hub_name','present_games','absent_games','status','with_hub_entropy','without_hub_entropy','entropy_change','equal_total_entropy_change']].to_string(index=False));return 0
if __name__=='__main__':raise SystemExit(main())
