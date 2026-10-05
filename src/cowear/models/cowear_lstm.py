#!/usr/bin/env python3
"""CoWear LSTM for common-target displacement and uncertainty prediction.

Jointly detected events only synchronize streams. Each device has its own
packed unidirectional LSTM and predicts the target displacement in its
gyro-local frame. Gyro supplies the rotation used to put candidates into world space.
At inference their Gaussian increment observations are fused by information
weighting; orientation uncertainty is added analytically to each covariance.
"""
from __future__ import annotations

import argparse, json, sys
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

from .._internal import event_pipeline as shared
from .. import DATASET_VERSION, INPUT_FEATURES, MODEL_HIDDEN_SIZE, PAPER_ROLE_FOR_STORAGE, PROTOCOL_VERSION
from ..protocol.checkpoints import read_checkpoint
from ..protocol.geometry import information_fusion
from ..training import fit_supervised

ROLES = shared.ROLES

@dataclass
class Item:
    x: np.ndarray; local_disp: np.ndarray; global_disp: np.ndarray; rotation: np.ndarray
    base_id: str; a: int; b: int; target_end: np.ndarray; same_hand: str

class CoWearEventDataset(Dataset):
    def __init__(self, sessions, split, role, fs, target_mode='mobile', input_frame='device', watch_hand='all'):
        self.items=[]
        for s in sessions:
            if s.split != split or (role=='watch' and watch_hand!='all' and str(s.same_hand)!=watch_hand): continue
            # Integrate W<-I with gyro in I, then form W<-G(t)=W<-I(t) I<-G(t).
            wi = np.empty((len(s.t),3,3),dtype=np.float32)
            wi[0] = Rotation.from_quat(s.quat[role][0]).as_matrix().astype(np.float32)
            gyro = s.raw_features[role][:,:3].astype(np.float64)
            for i in range(len(wi)-1):
                wi[i+1] = wi[i] @ Rotation.from_rotvec(gyro[i]/fs).as_matrix().astype(np.float32)
            if input_frame == 'device':
                rotation = wi
                features = s.raw_features[role][:, :INPUT_FEATURES]
            else:
                raise ValueError("CoWear LSTM uses device-frame gyro+accelerometer input")
            target_role = role if target_mode=='self' else target_mode
            for ar, br in s.boundaries:
                a,b=int(ar),int(br)
                if b<=a+3 or not s.edge_valid[role][a:b].all() or not s.edge_valid[target_role][a:b].all(): continue
                world = s.pos[target_role][b].astype(np.float32)-s.pos[target_role][a].astype(np.float32)
                self.items.append(Item(features[a:b+1].astype(np.float32),(rotation[a].T@world).astype(np.float32),world.astype(np.float32),rotation[a].astype(np.float32),s.base_id,a,b,s.pos[target_role][b].astype(np.float32),str(s.same_hand)))
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

def collate(batch):
    lens=torch.tensor([len(x.x) for x in batch]); x=torch.zeros(len(batch),int(lens.max()),INPUT_FEATURES)
    for i,v in enumerate(batch): x[i,:len(v.x)]=torch.from_numpy(v.x)
    return {'x':x,'lengths':lens,'local_disp':torch.from_numpy(np.stack([v.local_disp for v in batch])),
      'global_disp':torch.from_numpy(np.stack([v.global_disp for v in batch])),'rotation':torch.from_numpy(np.stack([v.rotation for v in batch])),'base_id':[v.base_id for v in batch], 'a':[v.a for v in batch], 'b':[v.b for v in batch], 'target_end':np.stack([v.target_end for v in batch]),'same_hand':[v.same_hand for v in batch]}

def cache_dataset(args,sessions,split,role,watch_hand='all'):
    p=args.cache_dir/f'{args.target_mode}_{args.input_frame}_nll_{role}_{watch_hand}_{split}.pt'
    meta={'v':8,'role':role,'split':split,'fs':args.sample_rate_hz,'target_mode':args.target_mode,'input_frame':args.input_frame,'input_features':INPUT_FEATURES,'watch_hand':watch_hand}
    if p.exists() and not args.rebuild_cache:
        q=torch.load(p,map_location='cpu',weights_only=False)
        if q.get('meta')==meta:
            d=CoWearEventDataset.__new__(CoWearEventDataset);d.items=[Item(**v) for v in q['items']];print(f'event_cache=hit role={role} split={split} events={len(d)}');return d
    d=CoWearEventDataset(sessions,split,role,args.sample_rate_hz,args.target_mode,args.input_frame,watch_hand);p.parent.mkdir(parents=True,exist_ok=True);torch.save({'meta':meta,'items':[asdict(x) for x in d.items]},p);print(f'event_cache=written role={role} hand={watch_hand} split={split} events={len(d)}');return d

def stats(data):
    z=np.concatenate([v.x for v in data.items]); return torch.tensor(z.mean(0),dtype=torch.float32),torch.tensor(z.std(0).clip(1e-4),dtype=torch.float32)

class CoWearLSTM(nn.Module):
    """Causal event encoder: local mean, globally supervised diagonal covariance."""
    def __init__(self,mean,std,hidden):
        super().__init__();self.register_buffer('feature_mean',mean.clone());self.register_buffer('feature_std',std.clone())
        self.input=nn.Sequential(nn.Linear(INPUT_FEATURES,hidden),nn.GELU());self.rnn=nn.LSTM(hidden,hidden,num_layers=2,batch_first=True,dropout=.1)
        self.local_head=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,3))
        self.global_logvar_head=nn.Sequential(nn.LayerNorm(hidden+9),nn.Linear(hidden+9,hidden),nn.GELU(),nn.Linear(hidden,3))
        nn.init.zeros_(self.local_head[-1].weight);nn.init.zeros_(self.local_head[-1].bias);nn.init.zeros_(self.global_logvar_head[-1].weight);nn.init.constant_(self.global_logvar_head[-1].bias,-2.)
    def forward(self,x,lengths,rotation):
        z=self.input((x-self.feature_mean)/self.feature_std);_,(h,_)=self.rnn(pack_padded_sequence(z,lengths.cpu(),batch_first=True,enforce_sorted=False));latent=h[-1]
        local=self.local_head(latent);global_mean=torch.einsum('bij,bj->bi',rotation,local);logvar=self.global_logvar_head(torch.cat((latent,rotation.flatten(1)),1)).clamp(-8,4)
        return {'local_mean':local,'global_mean':global_mean,'global_log_variance':logvar}

def losses(pred,local_target,global_target):
    local=F.smooth_l1_loss(pred['local_mean'],local_target,beta=.1)
    error=pred['global_mean']-global_target;lv=pred['global_log_variance'];nll=.5*(error.square()*torch.exp(-lv)+lv).mean()
    return local+nll,local,nll

def epoch(model,loader,device,opt=None):
    train=opt is not None; model.train(train); total=n=0
    for q in loader:
        x=q['x'].to(device);l=q['lengths'].to(device);local=q['local_disp'].to(device);global_target=q['global_disp'].to(device);rotation=q['rotation'].to(device)
        with torch.set_grad_enabled(train):
            loss,_,_=losses(model(x,l,rotation),local,global_target)
            if train: opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        total+=loss.item()*len(l);n+=len(l)
    return total/max(n,1)


def _epoch_step(model, loader, device, optimizer, _epoch):
    return epoch(model, loader, device, optimizer)

@torch.no_grad()
def infer(model,data,device,batch):
    out={};model.eval()
    for q in DataLoader(data,batch_size=batch,shuffle=False,collate_fn=collate):
        r=q['rotation'].to(device);p=model(q['x'].to(device),q['lengths'].to(device),r);mu=p['global_mean'].cpu().numpy();lv=p['global_log_variance'].cpu().numpy()
        for i,k in enumerate(zip(q['base_id'],q['a'],q['b'])):
            out[(k[0],int(k[1]),int(k[2]))]=(mu[i],np.diag(np.exp(lv[i])),q['global_disp'][i].numpy())
    return out

def evaluate(preds, args, allowed_sessions=None):
    report={}
    keys=sorted(k for k in set().union(*[set(v) for v in preds.values()]) if allowed_sessions is None or k[0] in allowed_sessions)
    methods=list(preds)+(['fused'] if args.target_mode!='self' else [])
    for name in methods:
        e=[]; maha=[]; count=0
        for k in keys:
            if name=='fused':
                av=[preds[r][k] for r in preds if k in preds[r]]
                m,c=information_fusion(np.stack([v[0] for v in av]),np.stack([v[1] for v in av])); target=av[0][2]
            elif k in preds[name]: m,_,target=preds[name][k]
            else: continue
            err=m-target;e.append(np.linalg.norm(err[[0,2]]));maha.append(float(err@np.linalg.solve(c if name=='fused' else preds[name][k][1]+np.eye(3)*1e-6,err)));count+=1
        # Roll out only successive joint events; gaps are never invented as zero motion.
        trajectory=[]
        for sid in sorted({k[0] for k in keys}):
            seq=[]
            for k in keys:
                if k[0] != sid: continue
                if name=='fused':
                    av=[preds[r][k] for r in preds if k in preds[r]]
                    m,_=information_fusion(np.stack([v[0] for v in av]),np.stack([v[1] for v in av])); target=av[0][2]
                elif k in preds[name]: m,_,target=preds[name][k]
                else: continue
                seq.append((k[1],k[2],m,target))
            seq.sort()
            pm=np.zeros(3); pt=np.zeros(3); previous=None
            for a,b,m,target in seq:
                if previous is not None and a != previous: pm=np.zeros(3);pt=np.zeros(3)
                pm += m; pt += target; trajectory.append(np.linalg.norm((pm-pt)[[0,2]])); previous=b
        report[name]={'events':count,'event_horizontal_rmse_m':float(np.sqrt(np.mean(np.square(e)))),'event_horizontal_mae_m':float(np.mean(e)),'mean_mahalanobis_sq_3d':float(np.mean(maha)),'rollout_horizontal_rmse_m':float(np.sqrt(np.mean(np.square(trajectory)))),'rollout_horizontal_mae_m':float(np.mean(trajectory))}
    return report

def fused_increment(preds,key):
    values=[preds[r][key] for r in preds if key in preds[r]]
    mean,covariance=information_fusion(np.stack([v[0] for v in values]),np.stack([v[1] for v in values]))
    return mean,values[0][2]

def main():
    p=argparse.ArgumentParser();
    p.add_argument('--processed-root',type=Path,required=True);p.add_argument('--split-index',type=Path,required=True);p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--roles',default='mobile,rokid,watch');p.add_argument('--target-mode',choices=('mobile','rokid','watch','self'),default='mobile');p.add_argument('--input-frame',choices=('device',),default='device');p.add_argument('--split-watch-hand',action='store_true');p.add_argument('--sample-rate-hz',type=float,default=100);p.add_argument('--hidden',type=int,default=MODEL_HIDDEN_SIZE);p.add_argument('--batch-size',type=int,default=1024);p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=5);p.add_argument('--min-delta',type=float,default=0.0);p.add_argument('--learning-rate',type=float,default=1e-3);p.add_argument('--gyro-noise-std',type=float,default=.015);p.add_argument('--gyro-bias-rw-std',type=float,default=.003);p.add_argument('--device',default='cuda');p.add_argument('--rebuild-cache',action='store_true');p.add_argument('--evaluate-only',action='store_true');p.add_argument('--resume-existing',action='store_true');p.add_argument('--calibration-mode',choices=('published_extrinsic','train_role_mean'),default='published_extrinsic');p.add_argument('--max-gap-s',type=float,default=.35);p.add_argument('--min-duration-s',type=float,default=4);p.add_argument('--smooth-s',type=float,default=.08);p.add_argument('--gravity-s',type=float,default=.55);p.add_argument('--truth-max-gap-s',type=float,default=.02);p.add_argument('--truth-max-jump-deg',type=float,default=45);p.add_argument('--truth-max-angular-rate-dps',type=float,default=2000);p.add_argument('--truth-max-speed-mps',type=float,default=15);p.add_argument('--imu-gap-floor-s',type=float,default=.03);p.add_argument('--watch-imu-gap-floor-s',type=float,default=.1);p.add_argument('--imu-gap-median-factor',type=float,default=2.5);p.add_argument('--imu-gap-iqr-factor',type=float,default=3);p.add_argument('--imu-gap-ceiling-s',type=float,default=.15);p.add_argument('--seed',type=int,default=2027)
    args=p.parse_args();shared.seed_everything(args.seed);args.output_dir.mkdir(parents=True,exist_ok=True);device=torch.device(args.device if torch.cuda.is_available() else 'cpu');sessions=shared.load_or_build_sessions(args);summary={}
    tasks=[('mobile','mobile','all'),('rokid','rokid','all'),('watch_same','watch','same'),('watch_different','watch','different')] if args.split_watch_hand else [(r,r,'all') for r in args.roles.split(',')]
    if not args.evaluate_only:
        for label,role,hand in tasks:
            tr=cache_dataset(args,sessions,'train',role,hand);va=cache_dataset(args,sessions,'val',role,hand);te=cache_dataset(args,sessions,'test',role,hand);mean,std=stats(tr);m=CoWearLSTM(mean,std,args.hidden).to(device);opt=torch.optim.AdamW(m.parameters(),lr=args.learning_rate,weight_decay=1e-4)
            checkpoint=args.output_dir/f'{label}.pt'
            if args.resume_existing and checkpoint.is_file():
                ck=read_checkpoint(checkpoint);m.load_state_dict(ck['state_dict']);best_epoch=int(ck.get('best_epoch',0));summary[label]={'best_epoch':best_epoch,'epochs_ran':best_epoch+args.patience,'best_val_total_loss':epoch(m,DataLoader(va,args.batch_size,False,collate_fn=collate),device),'test_total_loss':epoch(m,DataLoader(te,args.batch_size,False,collate_fn=collate),device),'resumed_checkpoint':True};print(f'resume model={label} best_epoch={best_epoch}',flush=True);continue
            fit = fit_supervised(
                m,
                DataLoader(tr,args.batch_size,True,collate_fn=collate),
                DataLoader(va,args.batch_size,False,collate_fn=collate),
                device,
                opt,
                _epoch_step,
                _epoch_step,
                args.epochs,
                args.patience,
                f'cowear_lstm/{label}',
                args.min_delta,
            )
            paper_target = PAPER_ROLE_FOR_STORAGE[role] if args.target_mode == 'self' else PAPER_ROLE_FOR_STORAGE[args.target_mode]
            torch.save({'state_dict':fit.best_state,'mean':mean,'std':std,'role':role,'watch_hand':hand,'best_epoch':fit.best_epoch,'metadata':{'model_version':'cowear_lstm_native6d_v1','role':PAPER_ROLE_FOR_STORAGE[role],'target':paper_target,'feature_frame':'device','input_features':INPUT_FEATURES,'split':'train_val_test_session_split','seed':args.seed,'dataset_version':DATASET_VERSION,'protocol_version':PROTOCOL_VERSION}},checkpoint);summary[label]={'best_epoch':fit.best_epoch,'epochs_ran':len(fit.history),'best_val_total_loss':fit.best_val_loss,'test_total_loss':epoch(m,DataLoader(te,args.batch_size,False,collate_fn=collate),device)}
    else:
        summary=json.loads((args.output_dir/'summary.json').read_text()) if (args.output_dir/'summary.json').exists() else {}
    predictions={}
    for label,role,hand in tasks:
        ck=read_checkpoint(args.output_dir/f'{label}.pt');model=CoWearLSTM(ck['mean'],ck['std'],args.hidden).to(device);model.load_state_dict(ck['state_dict']);values=infer(model,cache_dataset(args,sessions,'test',role,hand),device,args.batch_size)
        predictions.setdefault(role,{}).update(values)
    summary['configuration']={'target_mode':args.target_mode,'input_frame':args.input_frame,'early_stopping_patience':args.patience,'max_epochs':args.epochs,'split_watch_hand':args.split_watch_hand}
    summary['test_metrics']=evaluate(predictions,args)
    if args.split_watch_hand:
        hand_sessions={hand:{s.base_id for s in sessions if s.split=='test' and str(s.same_hand)==hand} for hand in ('same','different')}
        summary['test_metrics_by_watch_hand']={hand:evaluate(predictions,args,ids) for hand,ids in hand_sessions.items()}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
