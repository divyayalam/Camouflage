"""The final CAMouflage dashboard: live analysis and sourced experiment results."""
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import gradio as gr
from matplotlib import colormaps
from matplotlib.figure import Figure
import numpy as np

from analysis_core import (BASE, CLASSES, COLOR_VIEWS, as_image, color_ablate, gradcam, labeled,
                           load_model, model_choices, model_registry, paint_patch, patch_attention,
                           predict, preprocess_image, shortcut_reliance)
from channel_ablation import ablation_parts, parse_channels, predict_with_ablation
from dashboard_ui import CSS, THEME, INFERENCE_UI
from track1_results import add_track1_results_panel

UNKNOWN="Not specified"
ASSETS=BASE/"assets"

def confidence_plot(scores,names,title="Class confidence"):
    fig=Figure(figsize=(8,3.3),layout="constrained");ax=fig.subplots();x=np.arange(6)
    for i,(row,name) in enumerate(zip(scores,names)):
        ax.bar(x+(i-(len(scores)-1)/2)*.36,np.asarray(row)*100,.36,label=name,color=["#4f46e5","#0f9f94"][i%2])
    ax.set(xticks=x,xticklabels=CLASSES,ylim=(0,105),ylabel="Confidence (%)",title=title)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
    if len(names)>1:ax.legend(fontsize=8)
    return fig

def srs_plot(rows,names):
    fig=Figure(figsize=(7,3.2),layout="constrained");ax=fig.subplots();x=np.arange(2)
    for i,(row,name) in enumerate(zip(rows,names)):
        bars=ax.bar(x+(i-(len(rows)-1)/2)*.36,row,.36,label=name,color=['#4f46e5','#0f9f94'][i%2])
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    ax.set(xticks=x,xticklabels=['Color SRS','Patch SRS'],ylim=(0,115),ylabel='Score / 100')
    ax.spines[['top','right']].set_visible(False)
    if len(names)>1:ax.legend(fontsize=8)
    return fig

def analyze_image(image,key,actual,opacity):
    if actual not in [UNKNOWN,*CLASSES]:raise gr.Error("Choose an animal class or leave it unspecified.")
    if not isinstance(opacity,(float,int)) or not 0<=opacity<=1:raise gr.Error("Choose an overlay strength between 0 and 1.")
    x=preprocess_image(image);model=load_model(key);item=model_registry()[key]
    s=shortcut_reliance(model,x);ref=int(s['ref'][0]);scores=s['probs_clean'][0]
    wrong=(ref+1)%6
    if actual!=UNKNOWN:
        wrong=next((ref+k)%6 for k in range(1,6) if CLASSES[(ref+k)%6]!=actual)
    views=[x,color_ablate(x,'Grayscale'),color_ablate(x,'Swap BGR'),paint_patch(x,wrong)]
    names=['Clean','Grayscale','Red / blue swapped',f'{CLASSES[wrong].title()} patch']
    batch=np.concatenate(views);view_scores=predict(model,batch);maps=gradcam(model,batch);attention=patch_attention(maps)
    gallery=[];view_rows=[]
    for i,name in enumerate(names):
        informative=float(maps[i].max())>0
        color=colormaps['inferno'](maps[i])[...,:3]
        overlay=(1-opacity)*batch[i]+opacity*color if informative else batch[i]
        top=int(view_scores[i].argmax())
        label=f'{name} · {CLASSES[top]} {view_scores[i,top]:.1%}'
        if not informative:label+=' · no positive heatmap evidence'
        gallery.append((as_image(overlay).resize((256,256)),label))
        view_rows.append([name,CLASSES[top],round(float(view_scores[i,top])*100,2),round(float(view_scores[i,ref])*100,2),
                          round(float(attention[i]),2) if informative else None])
    tests=[]
    for name,row in s['colour_probs'].items():
        top=int(row[0].argmax());tests.append(['Color',name,CLASSES[top],round(float(row[0,top])*100,2),round(float(row[0,ref])*100,2),top!=ref])
    for k,row in s['patch_probs'].items():
        top=int(row[0].argmax());tests.append(['Patch',CLASSES[k]+' patch',CLASSES[top],round(float(row[0,top])*100,2),round(float(row[0,ref])*100,2),
                                              'Reference class · excluded' if k==ref else ('Follows patch' if top==k else 'Does not follow patch')])
    neutral=s['neutral_probs'][0];top=int(neutral.argmax())
    tests.append(['Control','Neutral gray patch',CLASSES[top],round(float(neutral[top])*100,2),round(float(neutral[ref])*100,2),bool(s['occlusion_changed'][0])])
    correctness='' if actual==UNKNOWN else f" · {'Correct' if CLASSES[ref]==actual else 'Incorrect'} for the supplied class **{actual}**"
    summary=(f"**{item['label']}** predicts **{CLASSES[ref]}** with **{scores[ref]:.2%}** confidence{correctness}.  \n"
             f"SRS follows this model’s clean prediction, **{CLASSES[ref]}**, for every cue test. Lower scores mean less sensitivity to these interventions.")
    report=dict(model=key,model_name=item['label'],actual_class=None if actual==UNKNOWN else actual,
                preprocessing='EXIF orientation; RGB; center square crop; bicubic resize 32x32; float32 / 255',
                reference=CLASSES[ref],confidence=float(scores[ref]),probabilities=labeled(scores),
                srs_colour=float(s['srs_colour'][0]),srs_patch=float(s['srs_patch'][0]),
                colour_conf_drop_pp=float(s['colour_conf_drop'][0]),patch_conf_drop_pp=float(s['patch_conf_drop'][0]),
                neutral_patch_changes_prediction=bool(s['occlusion_changed'][0]),
                srs_definition=dict(color='Percent of grayscale + five non-identity RGB permutations changing the clean prediction',
                                    patch='Percent of five patch classes other than the clean prediction which attract the prediction'),
                views=view_rows,interventions=tests)
    output=Path(tempfile.gettempdir())/'camouflage-reports'/uuid4().hex;output.mkdir(parents=True)
    path=output/'camouflage-image-report.json';path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return (labeled(scores),float(s['srs_colour'][0]),float(s['srs_patch'][0]),summary,as_image(x[0]),gallery,
            srs_plot([[float(s['srs_colour'][0]),float(s['srs_patch'][0])]],[item['label']]),tests,view_rows,
            gr.update(value=str(path),interactive=True))

def ablate_channels(image,key,text):
    model=load_model(key);x=preprocess_image(image)
    try:
        _,_,count,_=ablation_parts(model);channels=parse_channels(text,count)
        masked,layer=predict_with_ablation(model,x,channels)
    except ValueError as error:raise gr.Error(str(error)) from error
    before=predict(model,x)[0];after=masked[0];ref=int(before.argmax());top=int(after.argmax())
    message=(f"**{len(channels)} of {count} feature channels disabled** for this prediction. "
             f"{CLASSES[ref]} → {CLASSES[top]}. Confidence in {CLASSES[ref]}: **{before[ref]:.2%} → {after[ref]:.2%}** "
             f"({100*(after[ref]-before[ref]):+.2f} pp). Saved weights are unchanged.")
    return labeled(after),message,confidence_plot([before,after],['Channels on','Selected channels off'])

def compare_images(before_image,after_image,key,actual,mode,history):
    if mode not in ('Normal vs. conflict','Shortcut removal'):raise gr.Error('Choose a comparison type.')
    if actual not in [UNKNOWN,*CLASSES]:raise gr.Error('Choose an animal class or leave it unspecified.')
    if mode=='Shortcut removal' and actual==UNKNOWN:raise gr.Error('Select the true animal class for a shortcut-removal measurement.')
    batch=np.concatenate([preprocess_image(before_image),preprocess_image(after_image)])
    before,after=predict(load_model(key),batch);tops=np.argmax([before,after],axis=1)
    ref=CLASSES.index(actual) if actual!=UNKNOWN else int(tops[0]);delta=100*float(after[ref]-before[ref]);drop=-delta
    record=dict(number=len(history or [])+1,model=model_registry()[key]['label'],mode=mode,reference=CLASSES[ref],
                before_prediction=CLASSES[tops[0]],after_prediction=CLASSES[tops[1]],before=float(before[ref])*100,
                after=float(after[ref])*100,change=drop if mode=='Shortcut removal' else delta)
    records=[*(history or []),record]
    rows=[[r['number'],r['mode'],r['model'],r['reference'],r['before_prediction'],r['after_prediction'],round(r['before'],2),round(r['after'],2),round(r['change'],2)] for r in records]
    direction='drop · present minus removed' if mode=='Shortcut removal' else 'change · conflict minus normal'
    value=record['change']
    summary=(f"**{CLASSES[tops[0]]} → {CLASSES[tops[1]]}** · prediction {'changed' if tops[0]!=tops[1] else 'unchanged'}.  \n"
             f"Tracked class: **{CLASSES[ref]}** ({'true class' if actual!=UNKNOWN else 'normal-image prediction'}). "
             f"Confidence {direction}: **{value:+.2f} percentage points**.")
    return (labeled(before),labeled(after),record['before'],record['after'],value,summary,
            confidence_plot([before,after],['Before','After'],'Current pair'),history_plot(records),records,rows)

compare_pair=compare_images

def history_plot(records):
    fig=Figure(figsize=(7,3.3),layout='constrained');ax=fig.subplots();recent=records[-20:]
    ax.bar(np.arange(len(recent)),[r['change'] for r in recent],color='#4f46e5')
    ax.axhline(0,color='#94a3b8',lw=1);ax.set_ylabel('Reported difference (pp)');ax.set_xlabel('Comparison')
    ax.set_xticks(np.arange(len(recent)),[str(r['number']) for r in recent]);ax.spines[['top','right']].set_visible(False)
    ax.set_title('Session comparisons · latest 20')
    return fig

def compare_models(image,first,second,view,actual):
    if actual not in [UNKNOWN,*CLASSES]:raise gr.Error('Choose an animal class or leave it unspecified.')
    x=preprocess_image(image)
    if view=='Grayscale':x=color_ablate(x,'Grayscale')
    elif view=='Red / blue swapped':x=color_ablate(x,'Swap BGR')
    elif view=='Neutral gray patch':x=paint_patch(x)
    elif view!='Clean':raise gr.Error('Choose an input view.')
    names=[model_registry().get(key,{}).get('label',key) for key in (first,second)]
    stats=[shortcut_reliance(load_model(key),x) for key in (first,second)]
    scores=[item['probs_clean'][0] for item in stats];ref=CLASSES.index(actual) if actual!=UNKNOWN else int(scores[0].argmax())
    rows=[]
    for name,item,row in zip(names,stats,scores):
        top=int(row.argmax());rows.append([name,CLASSES[top],round(float(row[top])*100,2),round(float(row[ref])*100,2),
                                          round(float(item['srs_colour'][0]),2),round(float(item['srs_patch'][0]),2),
                                          '—' if actual==UNKNOWN else ('Yes' if CLASSES[top]==actual else 'No')])
    summary=(f"Both models receive the same **{view.lower()}** input. Compared confidence follows **{CLASSES[ref]}** "
             f"({'true class' if actual!=UNKNOWN else 'first model’s prediction'}). "
             f"Second minus first: **{100*float(scores[1][ref]-scores[0][ref]):+.2f} pp**. "
             "Each model’s SRS uses its own prediction as reference.")
    return (as_image(x[0]),*[labeled(row) for row in scores],summary,rows,confidence_plot(scores,names),
            srs_plot([[float(item['srs_colour'][0]),float(item['srs_patch'][0])] for item in stats],names))

def recorded():
    return json.loads((ASSETS/'notebook_results.json').read_text(encoding='utf-8'))

def recorded_pair(pair):
    return [[r['model'],round(r['mean_drop_pp'],2),round(r['std_drop_pp'],2),r['flips'],round(r['accuracy_with'],2),round(r['accuracy_without'],2)]
            for r in recorded()['paired'] if r['pair']==pair]

def recorded_figure(key):
    data=recorded();item=next(row for row in data['figures'] if row['key']==key)
    return str(ASSETS/item['file'])

def action(button,label,fn,inputs,outputs,api_name,controls=None,**options):
    """Lock the relevant controls until inference finishes, including validation failures."""
    controls=controls or inputs
    def begin():return [*[gr.update(interactive=False) for _ in controls],gr.update(value='Working…',interactive=False)]
    def finish():return [*[gr.update(interactive=True) for _ in controls],gr.update(value=label,interactive=True)]
    def checked(*args):
        try:return fn(*args)
        except gr.Error as error:
            # Validation is an ordinary user action, not a failed event chain.
            gr.Warning(getattr(error,'message',str(error)))
            return tuple(gr.skip() for _ in outputs)
    event=button.click(begin,outputs=[*controls,button],queue=False,show_progress='hidden',api_name=False)
    inference=event.then(checked,inputs=inputs,outputs=outputs,api_name=api_name,**INFERENCE_UI,**options)
    inference.then(finish,outputs=[*controls,button],queue=False,show_progress='hidden',api_name=False)
    return inference

def create_dashboard():
    choices=model_choices()
    if not choices:raise RuntimeError('Install at least one model checkpoint before launching CAMouflage.')
    default='run1_original' if 'run1_original' in dict((key,label) for label,key in choices) else choices[0][1]
    second='col_mixed_l12' if any(key=='col_mixed_l12' for _,key in choices) else 'g35_luma' if any(key=='g35_luma' for _,key in choices) else choices[-1][1]
    with gr.Blocks(title='CAMouflage · Understand the prediction',analytics_enabled=False) as demo:
        gr.HTML('<header id="camouflage-header"><div class="brand"><span class="brand-icon">C</span><div><h1>CAMouflage</h1><p>Understand the prediction. Test the shortcut.</p></div></div><span class="project-tag">IMAGE CLASSIFICATION &amp; EXPLAINABILITY</span></header>')
        with gr.Tabs(overflow_behavior='wrap'):
            with gr.Tab('Analyze Image',id='analyze'):
                gr.Markdown('### One image. A clearer explanation.\nSee the prediction, test its sensitivity to color and corner patches, and inspect the evidence with Grad-CAM.')
                with gr.Row():
                    with gr.Column(scale=5):
                        image=gr.Image(type='pil',label='Animal image',height=310,sources=['upload','clipboard'])
                        gr.Examples(examples=[[str(ASSETS/'examples'/f'{label}.png')] for label in CLASSES],inputs=image,label='Try a sample',examples_per_page=6)
                    with gr.Column(scale=4):
                        model=gr.Dropdown(choices=choices,value=default,label='Model')
                        actual=gr.Dropdown(choices=[UNKNOWN,*CLASSES],value=UNKNOWN,label='Actual animal · optional')
                        opacity=gr.Slider(0,1,value=.45,step=.05,label='Heatmap overlay strength')
                        gr.Markdown('Images are center-cropped and analyzed at **32 × 32** pixels. Predictions cover bird, cat, deer, dog, frog, and horse.')
                        run=gr.Button('Analyze image',variant='primary',size='lg')
                with gr.Row():
                    result=gr.Label(label='Prediction & confidence',num_top_classes=6)
                    color_srs=gr.Number(label='Color SRS / 100',precision=2,interactive=False)
                    patch_srs=gr.Number(label='Patch SRS / 100',precision=2,interactive=False)
                summary=gr.Markdown('Choose an image, then select **Analyze image**.')
                with gr.Accordion('How to read the scores',open=False):
                    gr.Markdown('**Color SRS:** the percentage of six color changes—grayscale and five channel permutations—that change the clean prediction. **Patch SRS:** the percentage of five other-class patches that attract the prediction to the patch’s class. Both range from 0 to 100; lower means less sensitivity to these probes. They measure cue sensitivity, not classification accuracy. No true label is needed. A neutral gray patch is a separate occlusion control.')
                gr.Markdown('### Visual evidence\nEach heatmap explains the prediction shown for that image version. Bright regions indicate positive evidence, not exact object boundaries.')
                gallery=gr.Gallery(label='Grad-CAM across image conditions',columns=(2,2,4,4,4,4),height=350,object_fit='contain',interactive=False,buttons=['download'])
                with gr.Accordion('Intervention details',open=False):
                    preview=gr.Image(label='Exact model input',interactive=False,height=180,buttons=['download'])
                    score_plot=gr.Plot(label='Shortcut Reliance Scores',format='png')
                    tests=gr.Dataframe(headers=['Probe','Intervention','Prediction','Top confidence (%)','Reference confidence (%)','Outcome'],value=[],interactive=False)
                    view_table=gr.Dataframe(headers=['View','Prediction','Confidence (%)','Reference confidence (%)','Corner attention × uniform'],value=[],interactive=False)
                    gr.Markdown('Corner attention compares the Grad-CAM mass in the top-left 4 × 4 region with that region’s area. **1×** is uniform attention. A blank value means there is no positive heatmap evidence.')
                download=gr.DownloadButton('Download image report',interactive=False)
                with gr.Accordion('Feature-channel ablation',open=False):
                    gr.Markdown('Temporarily turn off learned channels in the selected model’s final convolution block. This is a separate intervention from changing image colors or patches.')
                    channels=gr.Textbox(value='0, 1, 2',label='Channel IDs · 0–127',info='Comma-separated. Leave empty to compare with all channels enabled.')
                    mask=gr.Button('Apply channel mask')
                    mask_scores=gr.Label(label='Prediction after channel mask',num_top_classes=6)
                    mask_summary=gr.Markdown();mask_plot=gr.Plot(label='Channel ablation comparison',format='png')
                outputs=[result,color_srs,patch_srs,summary,preview,gallery,score_plot,tests,view_table,download]
                action(run,'Analyze image',analyze_image,[image,model,actual,opacity],outputs,'analyze_image',show_progress_on=[result,color_srs,patch_srs,gallery])
                def reset_analysis():
                    return None,None,None,'Choose an image, then select **Analyze image**.',None,[],None,[],[],gr.update(value=None,interactive=False),None,'',None
                for control in (image,model,actual,opacity):
                    control.input(reset_analysis,outputs=[*outputs,mask_scores,mask_summary,mask_plot],queue=False,show_progress='hidden',api_name=False)
                action(mask,'Apply channel mask',ablate_channels,[image,model,channels],[mask_scores,mask_summary,mask_plot],'ablate_channels')
                channels.input(lambda:(None,'',None),outputs=[mask_scores,mask_summary,mask_plot],queue=False,show_progress='hidden',api_name=False)

            with gr.Tab('Compare Images',id='pairs'):
                gr.Markdown('### Change the cue. Keep the animal.\nCompare two versions of the same image using one fixed model.')
                with gr.Row():
                    mode=gr.Radio(['Normal vs. conflict','Shortcut removal'],value='Normal vs. conflict',label='Comparison')
                    pair_model=gr.Dropdown(choices=choices,value=default,label='Model')
                    true=gr.Dropdown(choices=[UNKNOWN,*CLASSES],value=UNKNOWN,label='Actual animal')
                gr.Markdown('**Normal / conflict:** replace a cue with a misleading one. **Shortcut removal:** remove or neutralize the cue; select the actual class so confidence follows the true animal.')
                with gr.Row():
                    with gr.Column():before=gr.Image(type='pil',label='Normal / shortcut-present image',height=270,sources=['upload','clipboard']);before_scores=gr.Label(label='Before prediction',num_top_classes=6)
                    with gr.Column():after=gr.Image(type='pil',label='Conflict / shortcut-removed image',height=270,sources=['upload','clipboard']);after_scores=gr.Label(label='After prediction',num_top_classes=6)
                pair_run=gr.Button('Compare image pair',variant='primary')
                with gr.Row():
                    before_num=gr.Number(label='Tracked confidence · before (%)',precision=2,interactive=False)
                    after_num=gr.Number(label='Tracked confidence · after (%)',precision=2,interactive=False)
                    delta=gr.Number(label='Reported difference (pp)',precision=2,interactive=False)
                pair_summary=gr.Markdown('Normal/conflict reports **after − before**. Removal reports the confidence drop, **before − after**.')
                with gr.Row():pair_plot=gr.Plot(label='Current pair',format='png');session_plot=gr.Plot(label='Session comparisons',format='png')
                history=gr.State([])
                with gr.Accordion('Session history',open=False):
                    history_table=gr.Dataframe(headers=['#','Comparison','Model','Tracked class','Before prediction','After prediction','Before (%)','After (%)','Difference (pp)'],value=[],interactive=False)
                    clear=gr.Button('Clear comparison history')
                action(pair_run,'Compare image pair',compare_pair,[before,after,pair_model,true,mode,history],
                       [before_scores,after_scores,before_num,after_num,delta,pair_summary,pair_plot,session_plot,history,history_table],
                       'compare_images',controls=[before,after,pair_model,true,mode],concurrency_id='paired-history')
                clear.click(lambda:([],[],None),outputs=[history,history_table,session_plot],api_name='clear_comparison_history',concurrency_id='paired-history')
                def reset_pair():return None,None,None,None,None,'Choose a pair and select **Compare image pair**.',None
                for control in (before,after,pair_model,true,mode):
                    control.input(reset_pair,outputs=[before_scores,after_scores,before_num,after_num,delta,pair_summary,pair_plot],queue=False,show_progress='hidden',api_name=False)

            with gr.Tab('Compare Models',id='models'):
                gr.Markdown('### Compare the same image across models\nInspect predictions, class confidence, and SRS under identical inputs.')
                with gr.Row():
                    compare_image=gr.Image(type='pil',label='Animal image',height=280,sources=['upload','clipboard'])
                    with gr.Column():
                        first_model=gr.Dropdown(choices=choices,value=default,label='First model')
                        second_model=gr.Dropdown(choices=choices,value=second,label='Second model')
                        view=gr.Radio(['Clean','Grayscale','Red / blue swapped','Neutral gray patch'],value='Clean',label='Shared input view')
                        compare_true=gr.Dropdown(choices=[UNKNOWN,*CLASSES],value=UNKNOWN,label='Actual animal · optional')
                        compare_run=gr.Button('Compare models',variant='primary')
                with gr.Row():first_scores=gr.Label(label='First model',num_top_classes=6);second_scores=gr.Label(label='Second model',num_top_classes=6)
                comparison_summary=gr.Markdown('Choose an image and two models to compare.')
                comparison_table=gr.Dataframe(headers=['Model','Prediction','Top confidence (%)','Compared confidence (%)','Color SRS','Patch SRS','Correct?'],value=[],interactive=False)
                with gr.Row():comparison_plot=gr.Plot(label='Class confidence',format='png');comparison_srs=gr.Plot(label='Shortcut Reliance Scores',format='png')
                shared=gr.Image(label='Shared model input',interactive=False,height=180,buttons=['download'])
                action(compare_run,'Compare models',compare_models,[compare_image,first_model,second_model,view,compare_true],
                       [shared,first_scores,second_scores,comparison_summary,comparison_table,comparison_plot,comparison_srs],'compare_models')
                def reset_comparison():return None,None,None,'Choose an image and two models to compare.',[],None,None
                for control in (compare_image,first_model,second_model,view,compare_true):
                    control.input(reset_comparison,outputs=[shared,first_scores,second_scores,comparison_summary,comparison_table,comparison_plot,comparison_srs],queue=False,show_progress='hidden',api_name=False)

            with gr.Tab('Experiment Results',id='results'):
                data=recorded()
                gr.Markdown('### The final experiment\nRecorded results from Jay’s final notebook. These summarize held-out test images and stay fixed while you explore individual uploads.')
                with gr.Row():
                    gr.Number(value=data['n_test'],label='Images per accuracy condition',interactive=False,precision=0)
                    gr.Number(value=data['n_srs'],label='Images in the SRS study',interactive=False,precision=0)
                    gr.Number(value=data['n_gradcam'],label='Images in the attention study',interactive=False,precision=0)
                with gr.Tabs(overflow_behavior='wrap'):
                    with gr.Tab('Overview'):
                        gr.Markdown('**Color consistency training:** color SRS **25.4 → 1.2**. **Patch consistency training:** patch SRS **100.0 → 5.4**. These compare each family’s original model with its validation-selected mitigation.')
                        gr.Image(value=recorded_figure('mitigation'),label='Mitigation comparisons and controls',interactive=False,buttons=['download','fullscreen'])
                        rows=[[r['model'],r['acc_clean'],r['srs_colour'],r['srs_patch']] for r in data['models']]
                        gr.Dataframe(headers=['Model','Clean accuracy (%)','Color SRS','Patch SRS'],value=rows,interactive=False,label='Recorded model results')
                        gr.DownloadButton('Download recorded metrics',value=str(ASSETS/'recorded_results.csv'))
                    with gr.Tab('SRS & Robustness'):
                        for key in ('srs','accuracy','augmentation','confidence'):
                            gr.Image(value=recorded_figure(key),label=next(r['title'] for r in data['figures'] if r['key']==key),interactive=False,buttons=['download','fullscreen'])
                    with gr.Tab('Grad-CAM Study'):
                        gr.Markdown('Recorded examples compare the patch-trained model with patch consistency training, and the original color model with color consistency training. Heatmaps visualize relative evidence; they are not segmentation masks.')
                        for key in ('gradcam_patch','gradcam_color','attention'):
                            gr.Image(value=recorded_figure(key),label=next(r['title'] for r in data['figures'] if r['key']==key),interactive=False,buttons=['download','fullscreen'])
                    with gr.Tab('Confidence Study'):
                        gr.Markdown('**200 matched test images per model and pair.** Confidence follows the true animal class. Positive drops mean confidence fell without the cue. These measurements differ from the label-free SRS.')
                        pairs=[('Patch → neutral gray','patch_vs_neutral'),('Patch → clean image','patch_vs_removed'),('Color → grayscale','color_vs_grayscale'),('Color → BGR','color_vs_bgr')]
                        pair_choice=gr.Dropdown(choices=pairs,value='patch_vs_neutral',label='Recorded comparison')
                        paired_table=gr.Dataframe(headers=['Model','Mean drop (pp)','Std drop (pp)','Prediction flips','Accuracy with (%)','Accuracy without (%)'],value=recorded_pair('patch_vs_neutral'),interactive=False)
                        pair_choice.input(recorded_pair,inputs=pair_choice,outputs=paired_table,api_name='select_recorded_pair',show_progress='minimal')
                        with gr.Accordion('Open a saved confidence report',open=False):add_track1_results_panel()
                with gr.Accordion('Sources & methodology',open=False):
                    gr.Markdown('Recorded results and figures are preserved from the supplied **cifar10_animal_classification-3.ipynb**. Accuracy tables retain the notebook’s displayed precision; SRS means are shown at its printed one-decimal precision. The source notebook selected consistency winners on validation data. Live analysis uses the installed checkpoint named in its model selector; your local grayscale model is a separate training run from Jay’s recorded models. SRS uses the model’s clean prediction as reference, while the confidence study uses the true class.')
        gr.HTML('<footer id="camouflage-footer"><span>CAMouflage · GDG</span><span>Six animal classes · Live analysis &amp; recorded experiments</span></footer>')
    demo.queue(default_concurrency_limit=1,max_size=12)
    return demo
