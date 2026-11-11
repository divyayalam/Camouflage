"""Presentation shared by the final dashboard and report importer."""
import gradio as gr

THEME=gr.themes.Soft(primary_hue='indigo',secondary_hue='teal',neutral_hue='slate',
                      font=['Arial','sans-serif'],font_mono=['Consolas','monospace']).set(
    body_background_fill='#f5f7fb',body_background_fill_dark='#0f172a',
    block_radius='12px',button_large_radius='9px',button_small_radius='8px',
    block_border_width='1px',block_shadow='0 2px 10px rgba(15,23,42,.025)')
CSS='''
.gradio-container { width:100% !important; min-width:0 !important; box-sizing:border-box;
 max-width: 1240px !important; margin: 0 auto; padding: 26px 28px !important; }
#camouflage-header { display:flex; align-items:center; justify-content:space-between; gap:24px;
 padding:24px 28px; border:1px solid var(--border-color-primary); border-radius:16px;
 background:var(--block-background-fill); margin-bottom:18px; }
.brand { display:flex; align-items:center; gap:15px; }
.brand-icon { display:grid; place-items:center; width:48px; height:48px; border-radius:13px;
 background:#4f46e5; color:white; font-size:28px; font-weight:700; }
.brand h1 { font-size:27px; font-weight:750; letter-spacing:-.7px; margin:0; line-height:1.2; }
.brand p { color:var(--body-text-color-subdued); margin:6px 0 0; font-size:14px; }
.project-tag { color:#6366f1; font-size:10px; letter-spacing:1.2px; font-weight:700; text-align:right; }
[role="tablist"] { gap:6px; padding:6px; border:1px solid var(--border-color-primary) !important;
 border-radius:12px; background:var(--block-background-fill); margin-bottom:24px; }
[role="tab"] { border-radius:8px !important; padding:11px 18px !important; font-weight:600; }
[role="tab"][aria-selected="true"] { background:#eef2ff !important; color:#4338ca !important; border-bottom-color:transparent !important; }
[role="tabpanel"] { padding-top:4px !important; }
.gradio-container .image-frame img { object-fit:contain !important; }
.gradio-container .prose { line-height:1.65; }
.gradio-container .prose h3 { font-size:22px; letter-spacing:-.35px; margin-bottom:8px; }
.gradio-container .primary { min-height:46px; }
#camouflage-footer { display:flex; justify-content:space-between; gap:18px; margin-top:28px;
 padding:20px 4px 4px; border-top:1px solid var(--border-color-primary); font-size:12px; color:var(--body-text-color-subdued); }
@media (max-width:640px) {
 .gradio-container { padding:14px !important; }
 #camouflage-header { padding:18px; }
 .project-tag { display:none; }
 .brand h1 { font-size:23px; }
 [role="tab"] { padding:10px 11px !important; font-size:13px; }
 #camouflage-footer { flex-direction:column; gap:6px; }
}
'''
INFERENCE_UI=dict(show_progress='full',trigger_mode='once',concurrency_limit=1)
