"""Season-specific and franchise-clustered pooled summaries."""
from pathlib import Path
import json,subprocess,shutil
import numpy as np,pandas as pd
from scipy.stats import pearsonr,t
from hub_entropy_regressions import Fit,holm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def merge_pdf(files,target):
 import sys,io
 try:from pypdf import PdfReader,PdfWriter
 except ImportError:
  bundled=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages'
  if bundled.exists():sys.path.append(str(bundled))
  from pypdf import PdfReader,PdfWriter
 writer=PdfWriter()
 def label_pdf(width,height,label):
  label=label.replace('\\','\\\\').replace('(','\\(').replace(')','\\)');stream=f'BT /F1 11 Tf 14 {height-17:.2f} Td ({label}) Tj ET'.encode('ascii');objects=[b'<< /Type /Catalog /Pages 2 0 R >>',b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.2f} {height:.2f}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>'.encode(),b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>',b'<< /Length '+str(len(stream)).encode()+b' >>\nstream\n'+stream+b'\nendstream'];body=b'%PDF-1.4\n';offsets=[0]
  for i,obj in enumerate(objects,1):offsets.append(len(body));body+=f'{i} 0 obj\n'.encode()+obj+b'\nendobj\n'
  xref=len(body);body+=b'xref\n0 6\n0000000000 65535 f \n'+b''.join(f'{offset:010d} 00000 n \n'.encode() for offset in offsets[1:])+f'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF'.encode();return PdfReader(io.BytesIO(body)).pages[0]
 for file in files:
  reader=PdfReader(file);season=Path(file).parent.name
  for page in reader.pages:
   page.merge_page(label_pdf(float(page.mediabox.width),float(page.mediabox.height),'Season '+season));writer.add_page(page)
 with Path(target).open('wb') as stream:writer.write(stream)
 writer.close()

def report(root,out):
 root=Path(root);out=Path(out);frames=[];stats=[]
 for folder in sorted(out.iterdir()):
  f=folder/'team_entropy_and_ec.csv'
  if not f.exists() or not (folder/'summary.json').exists():continue
  d=pd.read_csv(f);d.insert(0,'season',folder.name);frames.append(d)
  exclusions=json.loads((folder/'tracking_exclusions.json').read_text()) if (folder/'tracking_exclusions.json').exists() else []
  if exclusions:
   method=folder/'METHOD_AND_RESULTS.md';body=method.read_text().split('\n\n## Tracking scope addendum')[0];note='\n\n## Tracking scope addendum\nUnverified tracking is not zero passes. Ambiguous multi-game windows may be replaced by smaller windows; only unresolved child windows are removed from the analyzed absence sample. The later-present reference is unavailable where full-period subtraction cannot be validated. Official and analyzed counts are in coverage_and_hubs.csv.\n\n'+json.dumps(exclusions,indent=2);method.write_text(body+note)
  s=pd.read_csv(folder/'ec_entropy_regressions.csv');s.insert(0,'season',folder.name);stats.append(s)
 if not frames:raise RuntimeError('No complete seasons; consult per-season run logs. No unavailable data are zero-filled.')
 data=pd.concat(frames,ignore_index=True)
 deleted=[]
 for z in data.itertuples():
  B=pd.read_csv(out/z.season/z.team/'deleted_training.csv',index_col=0).to_numpy();r=B.sum(1);nz=B>0;h=float(-(B[nz]/B.sum()*np.log2((B/np.maximum(r[:,None],1))[nz])).sum())/z.normalization_log2;deleted.append(h)
 data['excluded_later_absence_games']=data['excluded_later_absence_games'].fillna(0) if 'excluded_later_absence_games' in data else 0
 data['official_later_hub_absent_games']=data['official_later_hub_absent_games'].fillna(data.later_hub_absent_games) if 'official_later_hub_absent_games' in data else data.later_hub_absent_games
 data['later_present_reference_available']=data['later_present_reference_available'].fillna(data.entropy_later_present.notna()) if 'later_present_reference_available' in data else data.entropy_later_present.notna()
 data['entropy_simple_deletion']=deleted;data['mechanical_deletion_change']=data.entropy_simple_deletion-data.entropy_original;data['observed_minus_deleted_entropy']=data.entropy_actual-data.entropy_simple_deletion
 diagnostics=[]
 for season,d in data.groupby('season'):
  q=d[d.primary_eligible]
  for column in ['actual_change','mechanical_deletion_change','observed_minus_deleted_entropy']:
   r=pearsonr(q.ec_index,q[column]).statistic if len(q)>=4 else np.nan;diagnostics.append(dict(season=season,comparison=column,n=len(q),r=r,r_squared=r*r))
 pd.DataFrame(diagnostics).to_csv(out/'mechanical_baseline_diagnostics.csv',index=False)
 s=pd.concat(stats,ignore_index=True);s['holm_across_reported_season_tests']=holm(s.hc3_p);s.to_csv(out/'season_regressions.csv',index=False);data.to_csv(out/'all_team_season_entropy_and_ec.csv',index=False)
 coverage=data.groupby('season').agg(teams=('team','size'),observed_absence_pairs=('actual_passes',lambda x:sum(x>0)),eligible_teams=('primary_eligible','sum'),excluded_absence_games=('excluded_later_absence_games','sum'));coverage.to_csv(out/'season_coverage.csv')
 performance=[]
 for season,d in data.groupby('season'):
  q=d[d.primary_eligible];performance.append(dict(season=season,n=len(q),median_redistribution_edge_error=q.edge_distance_predicted_actual.median(),median_deletion_edge_error=q.edge_distance_deletion_actual.median(),teams_redistribution_improves=int((q.edge_distance_predicted_actual<q.edge_distance_deletion_actual-1e-8).sum()),median_redistribution_entropy_error=abs(q.entropy_actual-q.entropy_predicted).median(),median_deletion_entropy_error=abs(q.entropy_actual-q.entropy_simple_deletion).median()))
 performance=pd.DataFrame(performance);performance.to_csv(out/'prediction_performance_by_season.csv',index=False)
 pooled=[]
 for label,col,elig in [('primary_equal_count','actual_equal_count_change','primary_eligible'),('late_present_equal_count','concurrent_equal_count_change','concurrent_eligible')]:
  q=data[data[elig]].copy();x=q.ec_index.to_numpy();y=q[col].to_numpy();D=pd.get_dummies(q.season,drop_first=True,dtype=float).to_numpy();X=np.column_stack([np.ones(len(q)),x,D]);inv=np.linalg.inv(X.T@X);beta=inv@X.T@y;res=y-X@beta;meat=np.zeros((X.shape[1],X.shape[1]));groups=q.team.unique()
  for team in groups:
   take=q.team.eq(team).to_numpy();score=X[take].T@res[take];meat+=np.outer(score,score)
  G=len(groups);cov=G/(G-1)*(len(q)-1)/(len(q)-X.shape[1])*inv@meat@inv;se=np.sqrt(cov[1,1]);p=2*t.sf(abs(beta[1]/se),G-1);crit=t.ppf(.975,G-1);xx=q.ec_index-q.groupby('season').ec_index.transform('mean');yy=q[col]-q.groupby('season')[col].transform('mean');r=pearsonr(xx,yy).statistic
  pooled.append(dict(analysis=label,n=len(q),seasons=q.season.nunique(),franchises=G,season_adjusted_r=r,within_season_r_squared=r*r,slope=beta[1],slope_per_001=.01*beta[1],franchise_clustered_p=p,ci_low=beta[1]-crit*se,ci_high=beta[1]+crit*se))
 pooled_table=pd.DataFrame(pooled);pooled_table['holm_p_two_pooled_tests']=holm(pooled_table.franchise_clustered_p);pooled_table.to_csv(out/'pooled_season_adjusted_regressions.csv',index=False)
 with PdfPages(out/'all_seasons_regressions.pdf') as pdf:
  for analysis,col,elig in [('primary_equal_count','actual_equal_count_change','primary_eligible'),('late_present_equal_count','concurrent_equal_count_change','concurrent_eligible')]:
   fig,axes=plt.subplots(2,3,figsize=(17,10),sharex=True,sharey=True)
   for ax,(season,d) in zip(axes.flat,data.groupby('season')):
    q=d[d[elig]];ax.axhline(0,c='grey',lw=.6);ax.scatter(q.ec_index,q[col],s=26)
    for z in q.itertuples():ax.annotate(z.team,(z.ec_index,getattr(z,col)),fontsize=6,xytext=(2,2),textcoords='offset points')
    if len(q)>=4:
     model=Fit(q.ec_index.to_numpy(),q[col].to_numpy());grid=np.linspace(q.ec_index.min(),q.ec_index.max(),100);pred=model.prediction(grid);ax.plot(grid,pred['mean'],c='firebrick');ax.fill_between(grid,pred.mean_ci_lower,pred.mean_ci_upper,alpha=.1,color='firebrick');z=s[s.season.eq(season)&s.analysis.eq(analysis)].iloc[0];ax.set_title(f'{season}: n={len(q)}, r={z.r:+.3f}, R²={z.r_squared:.3f}\nHC3 p={z.hc3_p:.3f}',fontsize=11)
    else:ax.set_title(f'{season}: insufficient eligible teams')
    ax.set_xlabel('First-20 passing-only EC concentration');ax.set_ylabel('Actual absent − present entropy')
   for ax in axes.flat[data.season.nunique():]:ax.axis('off')
   fig.suptitle('First-20 passing hubs | '+analysis.replace('_',' '),fontsize=16);fig.tight_layout(rect=[0,.02,1,.94]);fig.savefig(out/(analysis+'_by_season.png'),dpi=150);pdf.savefig(fig);plt.close(fig)
  for result,column,eligibility in [(pooled[0],'actual_equal_count_change','primary_eligible'),(pooled[1],'concurrent_equal_count_change','concurrent_eligible')]:
   q=data[data[eligibility]].copy();q['centered_x']=q.ec_index-q.groupby('season').ec_index.transform('mean');q['centered_y']=q[column]-q.groupby('season')[column].transform('mean');fig,ax=plt.subplots(figsize=(12,8))
   for color,(season,d) in zip(plt.cm.tab10(np.linspace(0,1,q.season.nunique())),q.groupby('season')):ax.scatter(d.centered_x,d.centered_y,color=color,label=season,alpha=.65)
   grid=np.linspace(q.centered_x.min(),q.centered_x.max(),150);ax.plot(grid,result['slope']*grid,color='firebrick',label='Season-adjusted pooled fit');low=result['ci_low']*grid;high=result['ci_high']*grid;ax.fill_between(grid,np.minimum(low,high),np.maximum(low,high),color='firebrick',alpha=.1);ax.axhline(0,c='grey',lw=.6);ax.axvline(0,c='grey',lw=.6);ax.set_xlabel('EC concentration minus the eligible-team mean for that season');ax.set_ylabel('Entropy change minus the eligible-team mean for that season');ax.set_title(f"{result['analysis']} | season-adjusted pooled association\nn={result['n']}; within-season r={result['season_adjusted_r']:+.3f}, R²={result['within_season_r_squared']:.3f}\nFranchise-clustered p={result['franchise_clustered_p']:.4f}");ax.legend();fig.tight_layout();fig.savefig(out/(result['analysis']+'_pooled_adjusted.png'),dpi=150);pdf.savefig(fig);plt.close(fig)
  fig,axes=plt.subplots(1,2,figsize=(15,6));x=np.arange(len(performance))
  for ax,first,second,title in [(axes[0],'median_redistribution_edge_error','median_deletion_edge_error','Median edge-share total-variation error'),(axes[1],'median_redistribution_entropy_error','median_deletion_entropy_error','Median absolute normalized entropy error')]:
   ax.bar(x-.18,performance[first],.36,label='Role redistribution');ax.bar(x+.18,performance[second],.36,label='Simple deletion');ax.set_xticks(x,performance.season,rotation=30);ax.set_title(title);ax.set_ylabel('Lower means closer to actual absent network');ax.legend()
  fig.suptitle('Prediction checks | eligible teams only');fig.tight_layout();fig.savefig(out/'prediction_checks_by_season.png',dpi=150);pdf.savefig(fig);plt.close(fig)
  fig,ax=plt.subplots(figsize=(12,8));q=data[data.primary_eligible];colors=plt.cm.tab10(np.linspace(0,1,q.season.nunique()))
  for color,(season,d) in zip(colors,q.groupby('season')):ax.scatter(d.ec_index,d.actual_equal_count_change,color=color,label=season,alpha=.65)
  ax.axhline(0,c='grey');ax.set_xlabel('First-20 EC concentration');ax.set_ylabel('Actual absent − early hub-present entropy');ax.set_title('All eligible team-seasons | descriptive scatter\nPooled inference in CSV adjusts for season and clusters uncertainty by franchise');ax.legend();fig.tight_layout();fig.savefig(out/'pooled_scatter.png',dpi=150);pdf.savefig(fig);plt.close(fig)
 for filename,target in [('all_30_hub_networks.pdf','all_team_season_networks.pdf'),('entropy_changes_and_regressions.pdf','all_team_season_entropy_changes.pdf')]:
  files=[str(out/season/filename) for season in data.season.unique()]
  merge_pdf(files,out/target)
 quality=[]
 for season in data.season.unique():
  declared=json.loads((out/season/'tracking_exclusions.json').read_text()) if (out/season/'tracking_exclusions.json').exists() else [];excluded={(r['team'],r['kind'],r['start'],r['end']) for r in declared}
  for audit in (out/season/'passing_profiles').rglob('*.json'):
   info=json.loads(audit.read_text());frame=pd.read_csv(audit.with_suffix('.csv'));gp=float(frame.G.max()) if len(frame) and 'G' in frame else np.nan;quality.append(dict(season=season,team=info['team'],kind=info['kind'],player_id=info['player_id'],start=info['start'],end=info['end'],official_appearances=info['expected_games'],profile_G=gp,profile_G_below_official=bool(np.isfinite(gp) and gp<info['expected_games']),complete=info['complete'],used_in_analysis=(info['team'],info['kind'],info['start'],info['end']) not in excluded,zero_confirmation=info.get('zero_confirmation',''),error=info.get('error','')))
 quality=pd.DataFrame(quality);quality.to_csv(out/'profile_tracking_coverage.csv',index=False);quality.groupby(['season','kind']).agg(profiles=('complete','size'),complete=('complete','sum'),profile_G_below_official=('profile_G_below_official','sum')).to_csv(out/'profile_tracking_coverage_summary.csv')
 summary=dict(excluded_absence_games=int(data.excluded_later_absence_games.sum()),unavailable_later_present_references=int(data.later_present_reference_available.eq(False).sum()),seasons=data.season.unique().tolist(),team_seasons=len(data),eligible_team_seasons=int(data.primary_eligible.sum()),observed_pairs=int(data.actual_passes.gt(0).sum()),pooled=pooled);(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
 text=['# All-season first-20 passing-hub study',json.dumps(summary,indent=2),'Same study as the 2025–26 run: hub=max passes sent+received in first20; passing-only EC normalized HHI on A+A.T; predictions use only training rotations and previous-season NBA position metadata. Outgoing conditional entropy H(receiver|passer), weighted by each player’s outgoing pass share, with fixed within-team normalization. The reference denominator log2(N−1) uses the active union of training-present and observed absent players for retrospective evaluation. It does not alter frozen forecast edges; raw entropy in bits is also exported. The normalized forecast entropy therefore uses an evaluation reference, not a roster size known in advance. No minutes cutoff. Each panel sizes nodes by its own passing involvement and omits isolates.','2020–21 had 72 scheduled games per team, other saved seasons 82. All use first20 training and remaining observed eligible games, with retrospective trade censor as previously documented. At least15 early hub-present and10 later absent games for regression. No observed absence is unavailable, never zero entropy. Unresolved empty tracking windows are excluded as whole absence windows, not individual rows filled with zero; official and analyzed absence counts and dates are recorded. When exclusions prevent validating subtraction from full-season totals, the later-present reference is unavailable (NaN), and that team is omitted from that sensitivity fit. See tracking_exclusions.json and evaluation_tracking_scope.json in each affected season. No observed later-present games yields a known empty matrix and undefined reference entropy, rather than a tracking-data gap.','Primary outcome remains later absence minus first20 hub-present, equal-pass-count resampling. Later-present reference is sensitivity. Four fits per season, with Holm across all reported season tests in addition to within-season corrections. Thresholds and model choices are carried over from exploratory prior analysis.','Pooled fits include season indicators and franchise-clustered sandwich uncertainty with small-sample correction and t(G−1). Repeated franchise seasons are not treated as fully independent. Holm adjustment is also exported across the two pooled tests. The reported season-adjusted r and within-season R2 use season-demeaned x and y. They are not the R2 of a model including season indicators.','Limitations remain: first20 identity instability; missing broad position labels; arbitrary unvalidated replacement-role weights; roster mixing; opponent/co-absence confounding; shared network inputs and mechanical node-removal effects. More entropy is not evidence of improved performance. Neither aggregate differences nor regressions identify causal star dependency.','The 500 pass-count subsamples equalize count exposure; they are not 500 independent team observations or game-level uncertainty intervals. An additional descriptive diagnostic compares actual changes with simple deletion, and actual absent entropy minus deleted-network entropy. It is not a causal adjustment or a model of tactical adaptation. Complete retrieval audits do not guarantee that tracking recorded every pass in every game. profile_tracking_coverage.csv compares the passing-profile G field with official appearances: a lower G can reflect zero-activity games or incomplete tracking and is a flag, not proof of missing passes. Aggregate profiles cannot distinguish those explanations. Only official team participants within each query date window can be teammate recipients. Any discarded foreign or out-of-window recipient edges are listed in excluded_invalid_recipient_edges.csv, while raw profiles are retained. Full raw profiles, source snapshots, replacement allocations, frozen specifications, matrices, coverage and season-level reports are in each season folder. Previously validated 2025–26 results are reused unchanged; earlier seasons require independent date-filtered retrieval. Missing downloads are not treated as zero or replaced with full-season matrices.']
 (out/'METHOD_AND_RESULTS.md').write_text('\n\n'.join(text))
