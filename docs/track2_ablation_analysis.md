# Track 2 Feature/Channel Ablation Analysis

## Objective

Evaluate whether ablating candidate feature channels in the CAMouflage model weakens shortcut-dependent behavior while preserving performance on normal images.

## Method

Shortcut-sensitive examples were selected because the model originally classified them correctly but classified them incorrectly after a background swap. Predictions and average true-class confidence were compared before and after candidate-channel ablation for both normal images and their background-swapped counterparts.

The normal-image accuracy baseline of 100% and background-swap accuracy baseline of 0% are therefore **by construction**. These results describe the targeted shortcut-sensitive sample, not overall model accuracy on an unselected test set.

## Candidate Channels

The candidate channels used for ablation were:

`[109, 62, 105, 63, 31]`

## Results

| Metric on the targeted shortcut-sensitive sample | Before ablation | After ablation | Change |
| --- | ---: | ---: | ---: |
| Normal-image accuracy | 100% | 96% | −4 percentage points |
| Background-swap accuracy | 0% | 6% | +6 percentage points |
| Average true-class confidence, normal images | 76.86% | 76.38% | −0.48 percentage points |
| Average true-class confidence, background-swap images | 10.79% | 12.27% | +1.48 percentage points |

Ablation recovered **6 predictions** on background-swapped images.

## Recovered Examples

Examples of prediction changes among the recovered cases included:

- horse → dog
- frog → cat
- horse → cat
- horse → bird

Each arrow indicates the prediction before ablation followed by the recovered, correct prediction after ablation. These are illustrative examples, not a complete listing of all six recovered predictions.

## Interpretation

The results suggest that ablating the candidate channels weakens some shortcut-dependent behavior. Background-swap accuracy increased from 0% to 6%, and average true-class confidence on those images increased from 10.79% to 12.27%.

Most normal-image performance was preserved: accuracy remained at 96%, while average true-class confidence decreased slightly from 76.86% to 76.38%. The recovery came with a modest cost to normal-image accuracy on this selected sample.

These findings support a limited effect on shortcut dependence; they do not establish that the selected channels exclusively encode background shortcuts or that ablation eliminates shortcut reliance.

## Limitations

- **Selected sample:** The examples were originally correct and became incorrect after background swap. The 100%/0% baselines reflect this selection rule, so the results cannot be treated as overall test-set performance.
- **Center-region proxy masks:** Center-region masks are approximate foreground/background proxies. They may include background within the presumed foreground region or exclude parts of an object, limiting how cleanly background effects can be isolated.
- **Limited recovery rate:** Only 6 predictions were recovered, corresponding to 6% background-swap accuracy after ablation. Most swapped examples remained incorrect.
- **Normal-performance tradeoff:** Normal-image accuracy fell by 4 percentage points, indicating that ablation can also affect useful predictive information.

## Conclusion

Track 2 Week 6 demonstrated a working channel-ablation prototype using channels `[109, 62, 105, 63, 31]`. Ablation recovered six background-swap predictions and increased average true-class confidence on swapped images while preserving most normal-image performance. The results provide evidence of a partial reduction in shortcut-dependent behavior, with conclusions constrained by the targeted sample, approximate masks, and limited recovery rate.
