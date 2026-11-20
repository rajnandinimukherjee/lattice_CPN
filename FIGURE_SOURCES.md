# Paper figure sources

All generators use the semantic palette in `plot_settings.py`: dark gray for
undeformed data and variance-ratio-one guides; blue, orange, green, and purple
for direct variance and Approximations 1, 2, and 3 respectively. Figure 2's
deformed estimates use red shades, while training curves in Figures 7 and 8
are black.

Run commands from the repository root. `paper/main.tex` includes the 23 PDF files below. The plotting commands read saved results unless noted; run the corresponding analysis scripts first if their saved inputs are missing.

| PDF in `paper/figs/` | Generator |
| --- | --- |
| `xy_opf_locality_onept.pdf`, `xy_opf_locality_twopt.pdf` | `python xy_locality_comparison.py` |
| `xy_two_point_signal.pdf` | `python plot_xy_two_point_signal.py` |
| `xy_two_point_variance.pdf`, `xy_two_point_stn_legend.pdf` | `python plot_xy_two_point_stn.py` |
| `xy_radial_legend.pdf` | `python plot_xy_radial_profiles.py --legend-only --legend-output paper/figs/xy_radial_legend.pdf` |
| `csft_radial_profiles.pdf`, `xy_radial_profiles.pdf`, `cp2_radial_profiles.pdf` | `python plot_radial_profile_panels.py` |
| `xy_mask_radius_scan.pdf`, `xy_mask_radius_scan_legend.pdf` | `python plot_xy_mask_radius_scan.py` (uses JSON summaries from `xy_mask_radius_experiment.py`) |
| `csft_scaling_top.pdf`, `xy_scaling_top.pdf`, `cp2_scaling_top.pdf` | `python plot_opf_scaling_top.py` |
| `xy_field_comparison.pdf` | `python plot_xy_field_comparison.py` |
| `xy_approx2_training.pdf` | `python plot_xy_approx2_training.py` (reads the saved XY penalty training history and ensemble) |
| `locality_cpn_not_masked.pdf`, `locality_cpn_masked.pdf` | `python cpn_periodic_mask_comparison.py` |
| `opf_scaling_legend.pdf`, `csft_scaling_ratio.pdf`, `xy_scaling_ratio.pdf`, `cp2_scaling_ratio.pdf` | `python plot_appendix_opf_scaling.py` |
| `xy_physical_radius_profiles.pdf` | `python plot_xy_physical_radius_profiles.py` |

The two locality commands can load existing training artifacts, but may train when a compatible artifact is missing. The full plotting mode of `plot_xy_radial_profiles.py` writes exploratory figures to `plots/`; its legend command above writes the paper's shared legend.

Compile the draft with:

```shell
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```
