"""Inference, cue interventions, SRS and Grad-CAM from Jay's final pipeline."""
from functools import lru_cache
import json
import os
from pathlib import Path

import gradio as gr
import keras
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

CLASSES = ["bird", "cat", "deer", "dog", "frog", "horse"]
BASE = Path(__file__).resolve().parent
RUNS = BASE / "experiment_runs"
COLORS = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]], np.float32)
PATCH_SIZE = 4
COLOR_VIEWS = {"Grayscale": None, "Swap RBG": (0,2,1), "Swap GRB": (1,0,2),
               "Swap GBR": (1,2,0), "Swap BRG": (2,0,1), "Swap BGR": (2,1,0)}

def model_registry():
    records = {}
    def add(key, label, path, method):
        if path.is_file(): records[key] = dict(label=label, path=path, method=method)
    for run, label in (("run1_original","Original CNN"),("run2_mild","Mild augmentation"),
                       ("run3_stronger","Stronger augmentation"),("run4_heavy","Heavy augmentation")):
        add(run,label,RUNS/run/f"{run}_model.keras","Augmentation baseline")
    add("g35_luma","Grayscale augmentation · g35_luma",RUNS/"grayscale_local/g35_luma_only/g35_luma/g35_luma_model.keras","Grayscale augmentation")
    roots = [RUNS]
    external = os.environ.get("CAMOUFLAGE_MODELS_DIR")
    if external: roots.insert(0,Path(external).expanduser())
    for root in roots:
        for sc in (root/"shortcut_experiment",root/"jay_local/shortcut_experiment"):
            for level, name in (("run1_original","Patch-trained CNN"),("run4_heavy","Patch-trained · heavy augmentation")):
                add(level+"_shortcut",name,sc/f"{level}_shortcut.keras","Class-patch training")
        for family in ("consistency","grayscale"):
            folder=root/family
            if not folder.is_dir(): continue
            for directory in sorted(folder.iterdir()):
                if not directory.is_dir(): continue
                name=directory.name
                if family=="consistency":
                    description="mixed color" if name.startswith("col_mixed") else "grayscale" if name.startswith("col_gray") else "patch"
                    label=f"Consistency · {description} · {name.rsplit('_',1)[-1]}"
                else: label=f"Grayscale augmentation · {name}"
                add(name,label,directory/f"{name}_model.keras",family.title()+" training")
    return records

def model_choices():
    return [(item['label'],key) for key,item in model_registry().items()]

@lru_cache(maxsize=12)
def load_model(key):
    record=model_registry().get(key)
    if record is None: raise gr.Error("This model is not installed. Choose another model.")
    try:
        model=keras.models.load_model(record['path'],compile=False)
        if tuple(model.input_shape[1:])!=(32,32,3) or model.output_shape[-1]!=6:
            raise ValueError("Expected 32×32 RGB input and six animal classes")
        return model
    except (OSError,ValueError) as error:
        raise gr.Error("This model file could not be loaded. Choose another model or restore the checkpoint.") from error

def preprocess_image(image):
    if image is None: raise gr.Error("Upload an animal image to continue.")
    if not isinstance(image,Image.Image): raise gr.Error("Choose a valid image file.")
    image=ImageOps.exif_transpose(image).convert("RGB")
    w,h=image.size; size=min(w,h)
    image=image.crop(((w-size)//2,(h-size)//2,(w-size)//2+size,(h-size)//2+size))
    return np.asarray(image.resize((32,32),Image.Resampling.BICUBIC),np.float32)[None]/255.

def predict(model,x):
    scores=np.asarray(model(x,training=False))
    if scores.shape!=(len(x),6) or not np.isfinite(scores).all() or np.any(scores < -1e-6) or np.any(scores > 1+1e-6):
        raise gr.Error("The model returned invalid class scores.")
    return scores

def paint_patch(x,cls=None):
    result=x.copy()
    result[:,:PATCH_SIZE,:PATCH_SIZE]=.5 if cls is None else COLORS[cls]
    return result

def color_ablate(x,name):
    permutation=COLOR_VIEWS[name]
    return np.repeat(tf.image.rgb_to_grayscale(x).numpy(),3,-1) if permutation is None else x[...,list(permutation)].copy()

def shortcut_reliance(model,x):
    """Jay cell 48: reference is each model's own clean prediction, no true label."""
    views=[x,*[color_ablate(x,name) for name in COLOR_VIEWS],*[paint_patch(x,k) for k in range(6)],paint_patch(x)]
    probs=predict(model,np.concatenate(views)).reshape(14,len(x),6)
    clean=probs[0]; ref=clean.argmax(1); ar=np.arange(len(x)); c0=clean[ar,ref]
    color_probs=dict(zip(COLOR_VIEWS,probs[1:7])); patch_probs=dict(enumerate(probs[7:13]))
    changed=np.mean([p.argmax(1)!=ref for p in color_probs.values()],axis=0)
    other=[ref!=k for k in range(6)]
    hijack=np.sum([mask & (patch_probs[k].argmax(1)==k) for k,mask in enumerate(other)],axis=0)/5
    color_drop=np.mean([c0-p[ar,ref] for p in color_probs.values()],axis=0)
    patch_drop=np.sum([np.where(mask,c0-patch_probs[k][ar,ref],0) for k,mask in enumerate(other)],axis=0)/5
    return dict(ref=ref,ref_conf=c0,probs_clean=clean,colour_probs=color_probs,patch_probs=patch_probs,
                srs_colour=100*changed,srs_patch=100*hijack,colour_conf_drop=100*color_drop,
                patch_conf_drop=100*patch_drop,occlusion_changed=probs[13].argmax(1)!=ref,neutral_probs=probs[13])

@lru_cache(maxsize=12)
def gradient_model(model):
    conv=next((layer for layer in reversed(model.layers) if isinstance(layer,keras.layers.Conv2D)),None)
    if conv is None: raise gr.Error("This model does not support Grad-CAM.")
    return keras.Model(model.inputs[0],[conv.output,model.outputs[0]])

def gradcam(model,x,class_idx=None):
    """Jay cell 49: log-probability gradients, per-image spatial averaging."""
    view=gradient_model(model)
    with tf.GradientTape() as tape:
        activations,probs=view(tf.constant(x),training=False)
        indices=tf.argmax(probs,1,output_type=tf.int32) if class_idx is None else tf.constant(np.broadcast_to(class_idx,(len(x),)),tf.int32)
        target=tf.math.log(tf.gather(probs,indices,batch_dims=1)+1e-9)
    gradients=tape.gradient(target,activations)
    if gradients is None: raise gr.Error("A heatmap could not be generated for this model.")
    maps=tf.nn.relu(tf.reduce_sum(tf.reduce_mean(gradients,(1,2),keepdims=True)*activations,-1))
    maps=tf.image.resize(maps[...,None],(32,32),'bilinear')[...,0]
    maps=maps/(tf.reduce_max(maps,(1,2),keepdims=True)+1e-9)
    result=maps.numpy()
    if not np.isfinite(result).all(): raise gr.Error("A valid heatmap could not be generated for this image.")
    return result

def patch_attention(maps):
    share=maps[:,:PATCH_SIZE,:PATCH_SIZE].sum((1,2))/(maps.sum((1,2))+1e-9)
    return share/(PATCH_SIZE*PATCH_SIZE/(32*32))

def as_image(x):
    return Image.fromarray(np.rint(np.clip(x,0,1)*255).astype(np.uint8))

def labeled(scores): return dict(zip(CLASSES,map(float,scores)))
