# v14 recorded-result extension

Completed stages22–25 are supplied as originally distributed sources, with references, licenses and package identities preserved. `paper_assets/v14` supplies the completed current reports. Historical v12 checks and evidence remain in the unchanged v12 directory. New code changes here are limited to publication/table verification and figure rendering; detector inference remains the frozen experiment implementation.

The main manuscript now includes the ten-class COCO AP table (YOLO-1280, HyCADO-1280, SAHI-640 and both HyCADO differences). No results or model weights were tuned during publication. COCO and historical toolkit metrics are explicitly distinct. Confidence intervals cover fixed-model sample composition, not training variation. The split had already been examined; no independent new confirmation is claimed.

## Availability

- Historical code, source roles, metric records, stage22–25 code and current table/interval checks: included.
- Images/annotations: obtain from VisDrone maintainers; no new dataset-image distribution here.
- Twelve-example qualitative atlas: supplied with manuscript; this repository includes selection/export code, source identities and figure verification records.
- Checkpoints and full frozen predictions: hashes retained; public downloads pending.
- Full training replication from a clean clone: not validated.
- Detailed timing records: historical archive, separate from the current accuracy-focused manuscript.

Python 3.11+ and NumPy are sufficient for current recorded-result checks. Matplotlib is optional for figure regeneration. Full evaluation/export packages have separate original dependency and upstream-state requirements. Do not run GPU launchers against an active experiment directory without understanding those prerequisites.

The current extension manifest records hashes for every file added by this publication, excluding its own recursive identity. Existing historical materials and other repositories are not modified.

`paper_assets/v14/main.tex` is the reporting-source snapshot, not a standalone Overleaf build. Figures and the image atlas accompany the separate manuscript bundle. No dataset images are added.
