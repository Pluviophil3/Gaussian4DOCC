# Gaussian4DOCC

Gaussian Occupancy Prediction with Temporal Information Extraction and Spatiotemporal Fusion

### Abstract

In autonomous driving, occupancy prediction is an emerging and mainstream perception system that predicts the spatial occupancy and semantics of 3D voxel grids around the autonomous vehicle from image inputs. However, occupancy prediction generally requires dense occupancy representations with redundant computing. Although 3D Gaussian has recently been applied to improve occupancy prediction, the existing Gaussian occupancy predictions still have many issues, such as inefficient Gaussian representation, Gaussian distribution distortion, and sparse supervision signal. In this work, we propose a novel Gaussian occupancy prediction, called Gaussian4DOCC, to solve these issues by considering temporal information extraction and spatiotemporal fusion, which consists of three modules: 1) historical Gaussian fusion, 2) Gaussian feature aggregation, and 3) perspective supervision. We conducted experiments for an ablation study and compared Gaussian4DOCC to the stateof-the-art occupancy predictions on the well-known dataset nuScenes. The results show that the temporal information extraction and spatiotemporal fusion contribute to the improvement of occupancy prediction and our Gaussian4DOCC outperforms the existing occupancy predictions. Finally, we release the opensource code, video description, and datasets to facilitate Gaussian occupancy prediction in autonomous driving.

> [!CAUTION]
> **NON-STABLE VERSION WARN**
>
> This is under review. The content in this repository may be updated.

**IN THIS WORK**

- We propose a novel Gaussian fusion to merge the current Gaussian features with the historical Gaussian spatial distribution, which introduces prior Gaussian features to update the spatial updating scale for reducing inefficient Gaussians in empty regions and improving Gaussian representation efficiency. 
- We propose a novel Gaussian feature aggregation to alleviate the Gaussian distribution distortion in occlusion scenarios by fusing temporal information, which introduces a 4D spatiotemporal sampling of continuous image inputs and aggregates Gaussian query sequences to improve occupancy prediction in occlusion scenarios. 
- We propose perspective supervision as an auxiliary loss to improve the supervision density of the areas of interest during training, thus improving occupancy prediction accuracy.
