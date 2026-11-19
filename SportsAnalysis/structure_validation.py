"""Separated entropy/concentration, leader alignment, and exploratory absence tests."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import t
from entropy_structure_profiles import regress

MEASURES=[('passing_entropy','Passing recipient diversity'),('shot_category_entropy_bits','Shot-category diversity (bits)'),('raw_count_entropy','Pass + shot diversity (actual counts)')]
OUTCOMES=[('passing_ec_concentration','Passing EC concentration'),('offensive_action_hhi','Offensive workload concentration (HHI)')]
COLORS={'Shared leader':'#247c9c','Separate leaders':'#c56a37'}
def correct(frame,source='robust_p',target='holm_p'):
 order=frame[source].sort_values().index;frame.loc[order,target]=np.minimum(1,np.maximum.accumulate([(len(order)-j)*frame.loc[i,source] for j,i in enumerate(order)]));return frame

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--structure-validation',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path);a=p.parse_args(argv);root=a.output_dir;out=a.destination or root/'fg_attempt_networks/structure_validation';out.mkdir(parents=True,exist_ok=True)
 metrics=root/'fg_attempt_networks/raw_count_star_entropy_regressions/team_season_metrics.csv';profiles=root/'offensive_structure_profiles/team_season_profiles.csv'
 d=pd.read_csv(metrics).merge(pd.read_csv(profiles)[['team','season','passing_ec_concentration','offensive_action_hhi','leader_alignment','passing_leader_id','action_leader_id','passing_leader_name','action_leader_name','passing_leader_tie','action_leader_tie']],on=['team','season'],validate='one_to_one')
 assert len(d)==180 and np.allclose(d.ec_concentration,d.passing_ec_concentration,atol=1e-8) and np.allclose(d.usage_hhi,d.offensive_action_hhi)
 assert (d.leader_alignment.eq('Shared leader')==d.passing_leader_id.eq(d.action_leader_id)).all()
 d.to_csv(out/'metrics.csv',index=False);fits=[];interactions=[];sens=[];absence=[]
 with PdfPages(out/'structure_validation.pdf') as pdf:
  for x,title in MEASURES:
   fig,axes=plt.subplots(1,2,figsize=(15,6))
   for ax,(y,ylabel) in zip(axes,OUTCOMES):
    for group,g in [('All',d)]+list(d.groupby('leader_alignment')):
     fit,b,cov,_=regress(g,y,x=x);fit.update(group=group,analysis='main');fits.append(fit)
     if group=='All':continue
     color=COLORS[group];xx=np.linspace(g[x].min(),g[x].max(),100);X=np.column_stack([np.ones(len(xx)),xx-fit['entropy_mean']]);band=t.ppf(.975,fit['df'])*np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',X,cov,X),0));ax.scatter(g[x],g[y],c=color,s=25,alpha=.65);ax.plot(xx,X@b,c=color,label=f'{group}: n={len(g)}, r={fit["r"]:.3f}, R²={fit["r_squared"]:.1%}');ax.fill_between(xx,X@b-band,X@b+band,color=color,alpha=.12)
    fit,*_=regress(d,y,x=x,interaction='alignment');fit.update(analysis='alignment_interaction');interactions.append(fit)
    fit,*_=regress(d,y,x=x,season_fe=True);fit.update(analysis='season_FE');sens.append(fit)
    untied=d[~(d.passing_leader_tie|d.action_leader_tie)]
    fit,*_=regress(untied,y,x=x);fit.update(analysis='exclude_near_tied_leaders');sens.append(fit)
    ax.set_xlabel(title);ax.set_ylabel(ylabel);ax.legend(fontsize=8);ax.grid(alpha=.15)
   fig.suptitle(title+' vs separate concentration measures',fontsize=16);fig.text(.5,.015,'No minutes/roster controls. Separate regression lines by whether the same player leads EC and offensive workload.\nPooled inference and bands cluster by franchise; a formal slope-interaction test is exported. Associations share source data.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.075,1,.93]);fig.savefig(out/(x+'.png'),dpi=150);pdf.savefig(fig);plt.close(fig)
  file=root/'offensive_structure_profiles/absence_validation_coverage.csv'
  if file.exists():
   coverage=pd.read_csv(file);valid=coverage[coverage.eligible_for_exploratory_check.eq(True)].merge(d[['team','season','passing_entropy','shot_category_entropy_bits','raw_count_entropy']],on=['team','season'],validate='one_to_one');coverage.to_csv(out/'absence_coverage.csv',index=False);valid.to_csv(out/'absence_eligible_metrics.csv',index=False)
   fig,axes=plt.subplots(1,2,figsize=(14,6))
   for x in ['passing_ec_concentration','offensive_action_hhi','passing_entropy','shot_category_entropy_bits','raw_count_entropy']:
    fit,b,_,_=regress(valid,'adjusted_offensive_penalty',x=x);fit.update(status='retrospective selected absence sample; no causal validation');absence.append(fit)
    if x in dict(OUTCOMES):
     ax=axes[[v[0] for v in OUTCOMES].index(x)];ax.scatter(valid[x],valid.adjusted_offensive_penalty);xx=np.linspace(valid[x].min(),valid[x].max(),100);ax.plot(xx,b[0]+b[1]*(xx-fit['entropy_mean']),c='#b7432d');ax.axhline(0,color='grey',lw=.8);ax.set_xlabel(dict(OUTCOMES)[x]);ax.set_ylabel('Saved adjusted offensive absence penalty');ax.set_title(f'n={len(valid)}, r={fit["r"]:.3f}, R²={fit["r_squared"]:.1%}, p={fit["robust_p"]:.3f}')
   fig.suptitle('Exploratory absence check | compatible saved 2025–26 sample');fig.text(.5,.015,'Previously estimated penalties reused; not new adjustment for every injured teammate. Full-season predictors include absence games.\nSelected coverage, differing uncertainty and retrospective measurement prevent causal or external validation.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.08,1,.93]);fig.savefig(out/'absence_check.png',dpi=150);pdf.savefig(fig);plt.close(fig)
 correct(pd.DataFrame(fits)).to_csv(out/'regressions.csv',index=False)
 correct(pd.DataFrame(interactions),source='interaction_p').to_csv(out/'alignment_interactions.csv',index=False)
 pd.DataFrame(sens).to_csv(out/'sensitivity_models.csv',index=False)
 if absence:correct(pd.DataFrame(absence)).to_csv(out/'absence_regressions.csv',index=False)
 d.groupby(['season','leader_alignment']).size().unstack(fill_value=0).to_csv(out/'leader_counts.csv')
 sources=[metrics,profiles]+([file] if file.exists() else [])
 pd.DataFrame([dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sources]).to_csv(out/'source_manifest.csv',index=False)
 (out/'README.md').write_text('''Separate structural measures, no combined starness index in primary models. Passing EC normalized share HHI uses original A+A.T; workload HHI uses season player shares of FGA+0.44FTA+TOV, not official USG%. Shared leader means the same player ID leads both, deterministic tie rule with near-tie exclusion sensitivity. Entropies: normalized conditional passing recipient entropy; conditional choice entropy among three shot categories given shooter, in bits (maximum log2(3)); normalized actual-count passing+shot entropy. Shot-category variety is coarse, not shot efficiency, spatial diversity or total-offense entropy. Distinct shooters can use the same shot categories. Primary models unadjusted; franchise-clustered p/CIs, Holm within 18 fits, six interaction tests as separate family, five absence tests as separate family. Season FE and excluding near-tied leaders exported separately. Do not infer group difference from significance in only one group; use formal interactions. Raw r/R² in interaction/FE rows are unadjusted summaries; full_model_r_squared is full model fit.\n\nAbsence analysis reuses the prior selected 2025–26 adjusted penalty sample with matching hub IDs. It is retrospective and has only compatible teams; full-season predictors overlap absence windows. No fresh absence event study, opponent model or control for every simultaneously absent player is claimed. Positive penalty means higher offensive rating with the hub present. The saved within-team adjustment includes home game, opponent defensive rating excluding the current game, weighted absence of three other key teammates, game number and back-to-back status (nonvarying controls omitted). It only partially controls other player availability; no fresh penalty re-estimation was performed. It does not establish causal star dependency. Network null-model analysis is a separate script/output. All source hashes and group/sample counts exported.\n''')
 print(pd.DataFrame(fits)[['x','outcome','group','n','r','r_squared','robust_p']].to_string(index=False));print('Interactions:',pd.DataFrame(interactions)[['x','outcome','interaction_p']].to_string(index=False))
 if absence:print('Absence:',pd.DataFrame(absence)[['x','n','r','robust_p']].to_string(index=False))
 return 0
if __name__=='__main__':raise SystemExit(main())
