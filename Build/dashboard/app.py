"""Launch CAMouflage in the project's existing Python environment."""
from analysis_core import CLASSES as CLASS_NAMES, preprocess_image
from dashboard_ui import CSS, THEME
from final_dashboard import create_dashboard

demo=create_dashboard()

if __name__=='__main__':
    demo.launch(theme=THEME,css=CSS,ssr_mode=False,footer_links=[])
