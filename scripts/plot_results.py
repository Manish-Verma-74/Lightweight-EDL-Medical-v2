"""Generate reliability diagrams, risk-coverage curves and degradation plots from saved prediction bundles."""
import argparse
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from edl.cli import add_common_args, expected_run_identity, load_protocol  # noqa: E402
from edl.config import require_frozen, resolve  # noqa: E402
from edl.evaluation.metrics import bin_index, risk_coverage  # noqa: E402
from edl.evaluation.predict import load_bundle  # noqa: E402
from edl.training.checkpoint import check_identity  # noqa: E402
from edl.training.manifest import RunStatus  # noqa: E402


def reliability(probs, labels, bins=15):
    confidence = probs.max(1); correct = (probs.argmax(1) == labels).astype(float)
    idx = bin_index(confidence, bins); xs=[]; ys=[]; sizes=[]
    for b in range(bins):
        m = idx == b
        if m.any(): xs.append(float(confidence[m].mean())); ys.append(float(correct[m].mean())); sizes.append(int(m.sum()))
    return np.asarray(xs), np.asarray(ys), np.asarray(sizes)


def main():
    ap=argparse.ArgumentParser(); add_common_args(ap)
    ap.add_argument('--phase', default='primary'); ap.add_argument('--out-dir', default=None)
    ap.add_argument('--part', choices=['val','test'], default='test')
    ap.add_argument('--max-runs', type=int, default=None)
    args=ap.parse_args()
    if args.phase != 'primary': raise SystemExit('plots based on held-out test predictions are for the post-training primary evaluation only')
    cfg, split_df, run_df, split_sha, commit = load_protocol(args)
    require_frozen(cfg)
    root=os.path.join(args.drive_root,'primary','runs'); out=args.out_dir or os.path.join(args.drive_root,'results','plots')
    os.makedirs(out,exist_ok=True); test_ids=split_df.loc[split_df.split=='test','image_id'].astype(str).tolist()
    if args.max_runs: run_df=run_df.iloc[:args.max_runs]
    curve_csv=os.path.join(args.drive_root,'results','stats','corruption_curves.csv')
    curves=pd.read_csv(curve_csv) if os.path.exists(curve_csv) else None
    if curves is not None:
        if not {"config_hash", "split_sha256"}.issubset(curves.columns):
            raise ValueError("corruption_curves.csv lacks config/split identity columns")
        if not (curves["config_hash"].astype(str) == __import__("edl.config", fromlist=["config_hash"]).config_hash(cfg)).all():
            raise ValueError("corruption_curves.csv was generated from a different config hash")
        if not (curves["split_sha256"].astype(str) == split_sha).all():
            raise ValueError("corruption_curves.csv was generated from a different split manifest")
    for r in run_df.itertuples():
        rid=r.run_id; rd=os.path.join(root,rid); bundle=os.path.join(rd,'predictions_id.npz')
        if not os.path.exists(bundle): continue
        arrays,meta=load_bundle(bundle)
        expected=expected_run_identity(rid,cfg,split_sha,commit,'primary',False)
        check_identity(meta.get('identity',{}),expected,allow_commit_change=True)
        status=RunStatus(rd).read()
        check_identity(status.get('identity',{}),expected,allow_commit_change=True)
        eligible=status.get('state')=='completed' or (status.get('state')=='cap_limited' and status.get('cap_limited_at_resource_ceiling') is True)
        if not eligible: raise ValueError(f'{rid}: plots require an eligible primary run; state={status.get("state")}')
        if arrays[f'{args.part}_image_ids'].astype(str).tolist() != (test_ids if args.part=='test' else split_df.loc[split_df.split=='val','image_id'].astype(str).tolist()):
            raise ValueError(f'{rid}: {args.part} image order mismatch')
        probs=arrays[f'{args.part}_probs']; labels=arrays[f'{args.part}_labels']
        xs,ys,n=reliability(probs,labels,cfg['evaluation']['ece_bins'])
        run_out=os.path.join(out,rid); os.makedirs(run_out,exist_ok=True)
        fig,ax=plt.subplots(figsize=(5.2,4.6)); ax.plot([0,1],[0,1],linestyle='--',label='Perfect calibration')
        ax.plot(xs,ys,marker='o',label=f'{args.part}'); ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean confidence',ylabel='Empirical accuracy',title=f'Reliability: {rid}')
        ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(run_out,f'reliability_{args.part}.png'),dpi=150); plt.close(fig)
        errors=(probs.argmax(1)!=labels).astype(float)
        fig,ax=plt.subplots(figsize=(5.2,4.6))
        for key,unc in [('native',arrays[f'{args.part}_native_uncertainty']),('MSP',arrays[f'{args.part}_msp_uncertainty']),('entropy',arrays[f'{args.part}_entropy'])]:
            coverage,risk=risk_coverage(errors,unc); ax.plot(coverage,risk,label=key)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Coverage',ylabel='Selective risk',title=f'Risk–coverage: {rid}'); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(run_out,f'risk_coverage_{args.part}.png'),dpi=150); plt.close(fig)
        if curves is not None and (curves.run_id==rid).any():
            sub=curves[curves.run_id==rid].copy(); sub['condition']=sub['condition'].astype(str)
            # Conditions are emitted in frozen clean/noise/blur/brightness/contrast order.
            fig,ax=plt.subplots(figsize=(10,4.6)); x=np.arange(len(sub))
            ax.plot(x,sub['ece'],marker='.',label='ECE'); ax.plot(x,sub['ece_clean'],linestyle='--',label='Clean ECE')
            ax.set_xticks(x); ax.set_xticklabels(sub['condition'],rotation=75,ha='right',fontsize=7); ax.set_ylabel('ECE'); ax.set_title(f'Corruption calibration: {rid}'); ax.legend()
            fig.tight_layout(); fig.savefig(os.path.join(run_out,'corruption_ece_curve.png'),dpi=150); plt.close(fig)
            # Delta and its interval share the same zero-referenced axis; never overlay them on absolute ECE.
            fig,ax=plt.subplots(figsize=(10,4.6)); ax.axhline(0.0, linestyle='--', linewidth=1)
            ax.plot(x,sub['ece_delta'],marker='.',label='ECE change from clean')
            ax.fill_between(x,sub['ece_delta_ci_low'].to_numpy(),sub['ece_delta_ci_high'].to_numpy(),alpha=.18,label='95% bootstrap interval')
            ax.set_xticks(x); ax.set_xticklabels(sub['condition'],rotation=75,ha='right',fontsize=7); ax.set_ylabel('ECE change'); ax.set_title(f'Corruption ECE degradation: {rid}'); ax.legend()
            fig.tight_layout(); fig.savefig(os.path.join(run_out,'corruption_ece_delta_curve.png'),dpi=150); plt.close(fig)
            fig,ax=plt.subplots(figsize=(10,4.6)); ax.plot(x,sub['macro_f1'],marker='.',label='Macro-F1'); ax.plot(x,sub['macro_f1_clean'],linestyle='--',label='Clean Macro-F1')
            ax.set_xticks(x); ax.set_xticklabels(sub['condition'],rotation=75,ha='right',fontsize=7); ax.set_ylabel('Macro-F1'); ax.set_title(f'Corruption classification: {rid}'); ax.legend()
            fig.tight_layout(); fig.savefig(os.path.join(run_out,'corruption_macro_f1_curve.png'),dpi=150); plt.close(fig)
    print('plots saved to',out)

if __name__=='__main__': main()
