# Final dashboard validation

Validated November 10, 2026 against Jay's supplied final notebook.

## Implemented

- Final four-view dashboard: Analyze Image, Compare Images, Compare Models, Experiment Results.
- Exact final-notebook color and patch SRS definitions (cell 48), neutral-patch control, and reference-class confidence drops.
- Final-notebook log-probability Grad-CAM and area-normalized patch attention (cell 49).
- Center-square cropping and bicubic 32 × 32 preprocessing (cell 50).
- Live four-condition explanations, per-intervention tables, per-image JSON report, and channel ablation without weight mutations.
- Both paired-image modes, true-class removal confidence, retained-current-plot history clearing, and synchronized model comparisons.
- Twelve recorded model summaries, 52 paired-confidence summaries, and eight authentic embedded notebook figures. Source notebook SHA-256 and verified GitHub blob identity are in assets/notebook_results.json.
- Correct labels for grayscale/consistency models imported from the final Track 1 report schema.
- Consistent theme, responsive navigation/gallery, explicit loading states, disabled controls during inference, stale-result clearing, and recovery after invalid input.

## Verification

- 18 distinct held-out images, spanning all six classes, exercised the five installed checkpoints.
- SRS categorical results exactly match Jay's actual cell-48 function. Batched inference confidence differences are within 0.0002 percentage points; Grad-CAM maps match the actual cell-49 function within 1e-6.
- Tests cover RGB, grayscale and RGBA inputs, non-square center cropping, SRS zero/100 limits, exclusion of the reference patch class, multi-image indexing, and the separate neutral-patch control.
- Weight digests are unchanged after SRS, Grad-CAM, and ablation.
- Real Gradio callback processing covers analysis, channel masking, both paired-image modes, session clearing, model comparison, and recorded-pair selection.
- Browser testing completed 18 successive image analyses and exercised all four areas. The final visual/control regression passed after fixing narrow-layout overflow and empty-upload recovery.
- No unexpected browser console errors or HTTP responses with error status occurred in valid flows. The expected empty-upload notification was tested separately and all controls recovered.
- Report and CSV buttons initiate downloads; their endpoints return HTTP 200 and the expected contents. Automated native Chrome file saves were canceled by the test environment, so file contents were verified through the same download URLs. Native file-save behavior in the user's regular browser remains a manual check.
- Screenshots cover analysis, both comparison views, recorded results, SRS figures, and a 390-pixel viewport.

## Model availability

Jay-track1 was checked at 3ab4add74d68a34cc07cde1c454fa674cf07b0e1. It contains the final notebook and metrics sources, with no model checkpoints, exported final-evaluation files, or GitHub release assets. Main and sid-track1 contain the existing baseline/local-grayscale models and older augmentation archives, not the final consistency checkpoints.

Consequently, live analysis uses installed checkpoints, while the final consistency experiment is displayed from its authentic saved notebook outputs. The local g35_luma model and archived baseline weights are not represented as Jay's final g35_tone/consistency checkpoints. Dropping the actual exported final model folders into experiment_runs (or CAMOUFLAGE_MODELS_DIR) enables them after restart. No training or Git publication was performed.

## Run

Use the existing Camouflage Python environment to run app.py. No package changes were required.

## Evidence

See validation/final/ for the integration report, final browser regression report, and reviewed screenshots.
