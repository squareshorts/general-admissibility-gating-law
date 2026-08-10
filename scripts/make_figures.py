"""Generate the eight manuscript-facing figures from deterministic simulations and frozen empirical summaries."""
from __future__ import annotations
from pathlib import Path
import shutil, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; SIM=ROOT/'results'/'simulations'; EMP=ROOT/'results'/'empirical'; OUT=ROOT/'results'/'figures'; FINAL=ROOT/'figures'
OUT.mkdir(parents=True,exist_ok=True); FINAL.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'figure.dpi':160,'savefig.dpi':220,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def save(fig,stem):
    fig.savefig(OUT/f'{stem}.pdf',bbox_inches='tight'); fig.savefig(OUT/f'{stem}.png',bbox_inches='tight'); plt.close(fig)
def pivot(df,index,columns,values):
    t=df.pivot(index=index,columns=columns,values=values).sort_index().sort_index(axis=1); return t.columns.to_numpy(),t.index.to_numpy(),t.to_numpy()

def figure1():
    fig,ax=plt.subplots(figsize=(8.8,3.4)); ax.set_xlim(0,12.6); ax.set_ylim(0,4.9); ax.axis('off')
    face,edge='#f2f5fa','#3f5f8a'; tw,th,bw,bh=2.35,1.10,2.35,.96; yt,yb=3.55,1.75
    def box(cx,cy,w,h,title,sub):
        x0,y0=cx-w/2,cy-h/2; ax.add_patch(FancyBboxPatch((x0,y0),w,h,boxstyle='round,pad=0.03,rounding_size=0.07',linewidth=1.35,edgecolor=edge,facecolor=face)); ax.text(cx,cy+.16,title,ha='center',va='center',fontsize=11); ax.text(cx,cy-.20,sub,ha='center',va='center',fontsize=8.7,linespacing=1.05); return {'left':(x0,cy),'right':(x0+w,cy),'top':(cx,y0+h),'bottom':(cx,y0)}
    def arrow(a,b): ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',linewidth=1.35,color='#333333',shrinkA=4,shrinkB=4,mutation_scale=12))
    t=box(1.40,yt,tw,th,r'Truth state $T$','unobserved state'); p=box(4.35,yt,tw,th,r'Detector $P$','candidate-positive\ndecision'); q=box(7.30,yt,tw,th,r'Gate $Q$','admissibility rule'); a=box(10.25,yt,tw,th,r'$A^{+}$','actionable positive'); n=box(4.35,yb,bw,bh,r'$A^{-}$','explicit negative'); w=box(7.30,yb,bw,bh,r'$W$','withheld candidate')
    for x,y in ((t['right'],p['left']),(p['right'],q['left']),(q['right'],a['left']),(p['bottom'],n['top']),(q['bottom'],w['top'])): arrow(x,y)
    ax.text(6.30,.47,'A⁺ = P ∧ Q;   P = 1, Q = 0 → W;   W ≠ A⁻',ha='center',va='center',fontsize=10); save(fig,'figure1_decision_architecture')

def figure2():
    d=pd.read_csv(SIM/'phase_q1_q0.csv'); d=d[d.convention=='pure_abstention']; x,y,z=pivot(d,'retain_true_candidate','retain_false_candidate','ppv_change'); fig,ax=plt.subplots(figsize=(5.4,4.3)); lim=max(abs(np.nanmin(z)),abs(np.nanmax(z))); im=ax.pcolormesh(x,y,z,cmap='RdBu_r',norm=TwoSlopeNorm(vcenter=0,vmin=-lim,vmax=lim),shading='auto'); ax.plot([0,1],[0,1],'k--',lw=1.4,label=r'$q_1=q_0$'); ax.set(xlabel=r'False-candidate retention $q_0$',ylabel=r'True-candidate retention $q_1$'); fig.colorbar(im,ax=ax,label=r'PPV$_G$ − PPV$_U$'); ax.legend(frameon=False); save(fig,'figure2_ppv_boundary')

def figure3():
    d=pd.read_csv(SIM/'phase_q1_q0.csv'); d=d[d.convention=='abstention_plus_missed']; x,y,z=pivot(d,'retain_true_candidate','retain_false_candidate','delta_loss'); fig,ax=plt.subplots(figsize=(5.4,4.3)); lim=max(abs(z.min()),abs(z.max())); im=ax.pcolormesh(x,y,z,cmap='RdBu_r',norm=TwoSlopeNorm(vcenter=0,vmin=-lim,vmax=lim),shading='auto'); q0=np.linspace(0,1,400); pi,s,f,q1w,w0,cfp=.05,.85,.12,5,1,10; q1=1-((1-pi)*f*(1-q0)*(cfp-w0))/(pi*s*q1w); m=(q1>=0)&(q1<=1); ax.plot(q0[m],q1[m],'k-',lw=2,label=r'exact $\Delta_L=0$'); ax.set(xlabel=r'False-candidate retention $q_0$',ylabel=r'True-candidate retention $q_1$'); fig.colorbar(im,ax=ax,label=r'$\Delta_L$ (blue beneficial; red harmful)'); ax.legend(frameon=False); save(fig,'figure3_expected_loss_phase')

def figure4():
    d=pd.read_csv(SIM/'score_strategy_results.csv'); keep=['retuned_s_threshold','tuned_hard_conjunctive_gate','joint_s_z_model','population_bayes_s_z']; d=d[d.strategy.isin(keep)]; t=d.pivot(index='evidence_environment',columns='strategy',values='expected_loss').rename(index={'complementary_information':'Complementary information','conditionally_independent':'Conditionally independent','positively_correlated':'Positively correlated','shared_measurement_error':'Shared measurement error','strongly_redundant':'Strongly redundant'},columns={'joint_s_z_model':'Joint S,Z model','population_bayes_s_z':'Population Bayes S,Z','retuned_s_threshold':'Retuned S threshold','tuned_hard_conjunctive_gate':'Tuned hard gate'}); fig,ax=plt.subplots(figsize=(9.2,4.2)); t.plot.bar(ax=ax); ax.set(xlabel='Evidence structure',ylabel='Test expected loss'); ax.legend(frameon=False,fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1)); ax.tick_params(axis='x',rotation=20); save(fig,'figure4_evidence_policy_comparison')

def figure5():
    d=pd.read_csv(SIM/'finite_sample_summary.csv'); near=d[d.scenario=='near_boundary']; rare=d[d.scenario=='rare_false_retention']; x1,y1,z1=pivot(near,'prevalence','n','wrong_loss_sign_rate'); x2,y2,z2=pivot(rare,'prevalence','n','q0_zero_rate'); fig,axes=plt.subplots(1,2,figsize=(8,3.5)); im=axes[0].imshow(z1,origin='lower',aspect='auto',vmin=0,vmax=.5,cmap='magma'); axes[0].set(xticks=range(len(x1)),xticklabels=x1.astype(int),yticks=range(len(y1)),yticklabels=[f'{v:g}' for v in y1],xlabel='Sample size',ylabel='Prevalence'); axes[0].text(.03,.97,'(A) Sign-error frequency\nnear expected-loss boundary',transform=axes[0].transAxes,ha='left',va='top',weight='bold',fontsize=8.5,color='white'); fig.colorbar(im,ax=axes[0],pad=.03,label='Sign-error frequency'); im=axes[1].imshow(z2,origin='lower',aspect='auto',vmin=0,vmax=.5,cmap='viridis'); axes[1].set(xticks=range(len(x2)),xticklabels=x2.astype(int),yticks=range(len(y2)),yticklabels=[f'{v:g}' for v in y2],xlabel='Sample size'); axes[1].text(.03,.97,'(B) Observed $\\widehat q_0=0$\nwith small false retention',transform=axes[1].transAxes,ha='left',va='top',weight='bold',fontsize=8.5,color='white'); fig.colorbar(im,ax=axes[1],pad=.03,label='Observed-zero frequency'); save(fig,'figure5_finite_sample_uncertainty')

def figure6():
    u=pd.read_csv(EMP/'bootstrap_intervals.csv'); systems=['Challenge 2015','PTB-XL']; z=u.set_index(['dataset','metric']); rows=[z.loc[(s,'delta_q')] for s in systems]; est=np.array([r.estimate for r in rows]); lo=np.array([r.lower for r in rows]); hi=np.array([r.upper for r in rows]); y=np.arange(2)[::-1]; fig,ax=plt.subplots(figsize=(7.2,2.25)); ax.errorbar(est,y,xerr=np.vstack([est-lo,hi-est]),fmt='o',capsize=4,markersize=7,linewidth=1.7); ax.axvline(0,color='black',lw=.9,ls='--'); ax.set_yticks(y,systems); ax.set_xlabel(r'Candidate-conditioned discrimination, $\Delta_Q=q_1-q_0$'); [ax.annotate(f'{e:.4f} [{l:.4f}, {h:.4f}]',(e,yy),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8) for e,l,h,yy in zip(est,lo,hi,y)]; save(fig,'figure6_empirical_delta_q')

def figure7():
    c=pd.read_csv(EMP/'strategy_controls.csv'); c=c[(c.dataset=='PTB-XL')&c.strategy.isin(['original_candidate','hard_admissibility_gate','retuned_s_threshold','joint_s_q_model'])]; names={'original_candidate':'Original','hard_admissibility_gate':'Hard gate','retuned_s_threshold':'Retuned S','joint_s_q_model':'Joint (S,Q)'}; x=np.arange(len(c)); w=.26; fig,ax=plt.subplots(figsize=(7.5,3.8)); ax.bar(x-w,c.action_ppv,w,label='PPV'); ax.bar(x,c.operational_sensitivity,w,label='Sensitivity'); ax.bar(x+w,c.false_action_rate,w,label='False-action rate'); ax.set_xticks(x,[names[s] for s in c.strategy]); ax.set_ylim(0,.8); ax.set_ylabel('Proportion'); ax.legend(frameon=False,ncol=3); save(fig,'figure7_empirical_policy_comparison')

def figure8():
    d=pd.read_csv(EMP/'transport_strata.csv'); d=d[np.isfinite(d.delta_q)].copy();
    def label(r):
        v=' '.join(str(r.stratum).replace('_',' ').split()); return f"Challenge 2015 — {v} alarms" if r.stratum_variable=='alarm_type' else (f'PTB-XL — Device {v}' if r.stratum_variable=='device' else f'PTB-XL — Site {int(float(v))}')
    labels=d.apply(label,axis=1); fig,ax=plt.subplots(figsize=(8.8,max(4,.28*len(d)))); y=np.arange(len(d)); ax.scatter(d.delta_q,y,s=42); ax.axvline(0,color='black',lw=.8); ax.set_yticks(y,labels,fontsize=7); ax.set_xlabel(r'$q_1-q_0$'); save(fig,'figure8_operational_heterogeneity')

def main():
    for f in (figure1,figure2,figure3,figure4,figure5,figure6,figure7,figure8): f()
    for stem in ('figure1_decision_architecture','figure2_ppv_boundary','figure3_expected_loss_phase','figure4_evidence_policy_comparison','figure5_finite_sample_uncertainty','figure6_empirical_delta_q','figure7_empirical_policy_comparison','figure8_operational_heterogeneity'): shutil.copyfile(OUT/f'{stem}.pdf',FINAL/f'{stem}.pdf')
    print('Generated eight manuscript-facing PDF figures in figures/.')
if __name__=='__main__': main()
