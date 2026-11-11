# Usage: python verify_ablation.py <model.keras> <test_set.npz>   (npz needs keys images_present [N,32,32,3 in 0-1] and labels [N] ints 0-5)
# Needs channel_ablation.py (copied from Build/dashboard/) in the same folder.
import os, sys, json; os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, keras
import channel_ablation as ca
d=np.load(sys.argv[2]); X=d["images_present"].astype("float32"); Y=d["labels"]
M=sys.argv[1]
model=keras.models.load_model(M)
head,tail,count,lname=ca.ablation_parts(model); print("split at",lname,"channels",count)
base=model.predict(X,verbose=0,batch_size=500); pred=base.argmax(1); print("baseline acc (JPEG mirror)",round((pred==Y).mean(),4))
out={}
# ---- A. unit checks on the dashboard implementation ----
sub=X[:64]
p0,_=ca.predict_with_ablation(model,sub,[]);               out["empty_ids_max_diff_vs_model"]=float(np.abs(p0-model.predict(sub,verbose=0)).max())
pa,_=ca.predict_with_ablation(model,sub,[109,62,105,63,31]); out["ablation_changes_probs"]=float(np.abs(pa-p0).max())
pb,_=ca.predict_with_ablation(model,sub,[31,63,105,62,109]); out["order_invariant_max_diff"]=float(np.abs(pa-pb).max())
allc,_=ca.predict_with_ablation(model,X[:200],list(range(count))); out["all_channels_off_unique_outputs"]=int(len(np.unique(allc.round(6),axis=0)))
f=head(sub[:8],training=False).numpy(); mask=np.ones((1,1,1,count),np.float32); mask[...,[109,62,105,63,31]]=0
out["ablated_channels_are_zero"]=bool(np.all((f*mask)[...,[109,62,105,63,31]]==0))
# notebook-style implementation (ablate_channels on numpy copy, then classifier) must equal dashboard version
nb=f.copy(); nb[:,:,:,[109,62,105,63,31]]=0; pnb=tail(nb,training=False).numpy(); out["notebook_vs_dashboard_max_diff"]=float(np.abs(pnb-pa[:8]).max())
try: ca.predict_with_ablation(model,sub,[count]); out["out_of_range_raises"]=False
except ValueError: out["out_of_range_raises"]=True
# ---- B. replicate notebook experiment (class-sorted JPEG mirror, same seeds/code) ----
mask2=np.zeros((32,32,1),np.float32); mask2[8:24,8:24,:]=1
np.random.seed(42); n=len(X); partner=np.zeros(n,int)
for i in range(n): partner[i]=np.random.choice(np.where(Y!=Y[i])[0])
swap=X*mask2+X[partner]*(1-mask2)
sp=model.predict(swap,verbose=0,batch_size=500); spred=sp.argmax(1)
out["swap_acc"]=float((spred==Y).mean())
changed=np.where((pred==Y)&(spred!=Y))[0]; out["n_changed"]=int(len(changed))
Fo=head.predict(X,verbose=0,batch_size=500); Fs=head.predict(swap,verbose=0,batch_size=500)
def run(idx,chs):
    m=np.ones((1,1,1,count),np.float32); m[...,list(chs)]=0
    po=tail.predict(Fo[idx]*m,verbose=0,batch_size=500); ps=tail.predict(Fs[idx]*m,verbose=0,batch_size=500)
    y=Y[idx]; return dict(norm_acc=float((po.argmax(1)==y).mean()),swap_acc=float((ps.argmax(1)==y).mean()),
      swap_conf=float(ps[np.arange(len(y)),y].mean()),norm_conf=float(po[np.arange(len(y)),y].mean()))
# channel ranking exactly as notebook (first 100 changed examples, mean |swap - orig| of spatial-mean activation)
ev=changed[:100]
conv_act=keras.Model(model.inputs,[l for l in model.layers if isinstance(l,keras.layers.Conv2D)][-1].output)
ch=np.mean(np.abs(conv_act.predict(swap[ev],verbose=0).mean((1,2))-conv_act.predict(X[ev],verbose=0).mean((1,2))),0)
top=np.argsort(ch)[::-1][:10].tolist(); out["mirror_top10_channels"]=top
out["overlap_with_team_top5"]=sorted(set(top[:5])&{109,62,105,63,31})
team=[109,62,105,63,31]
out["team_channels_first100"]=run(ev,team); out["mirror_top5_first100"]=run(ev,top[:5])
held=changed[100:700]; out["team_channels_heldout600"]=run(held,team); out["mirror_top5_heldout600"]=run(held,top[:5]); out["no_ablation_heldout600"]=run(held,[])
# random-channel control on the held-out set
rng=np.random.default_rng(0); rs=[]
for _ in range(300):
    c=rng.choice(count,5,replace=False); r=run(held,c); rs.append((r["swap_acc"],r["norm_acc"],r["swap_conf"]))
rs=np.array(rs); out["random5_heldout_swap_acc_mean_p5_p95"]=[float(rs[:,0].mean()),float(np.percentile(rs[:,0],5)),float(np.percentile(rs[:,0],95))]
out["random5_heldout_norm_acc_mean_p5_p95"]=[float(rs[:,1].mean()),float(np.percentile(rs[:,1],5)),float(np.percentile(rs[:,1],95))]
out["random5_heldout_swap_conf_mean_p5_p95"]=[float(rs[:,2].mean()),float(np.percentile(rs[:,2],5)),float(np.percentile(rs[:,2],95))]
for k in ("team_channels_heldout600","mirror_top5_heldout600"):
    out[k+"_swap_acc_percentile_vs_random"]=float((rs[:,0]<out[k]["swap_acc"]).mean())
json.dump(out,open("verify_ablation_results.json","w"),indent=1)
print(json.dumps(out,indent=1))
