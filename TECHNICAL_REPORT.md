# 6D Object Pose Estimation with PointNet and Guarded ICP

## 1. Introduction

This project estimates an object's camera-frame 6D pose from RGB-D, an instance mask, camera intrinsics, and a canonical mesh. The final pipeline uses a learned PointNet pose initializer followed by guarded ICP refinement. ICP is accepted only when it improves geometric alignment without making an implausibly large pose change.

### Problem formulation

For each visible object instance, the input is an RGB image $I$, depth image $D$, known camera intrinsics $K$, instance mask $M$, object class $o$, and its canonical mesh $\mathcal{M}_o\subset\mathbb{R}^3$. The mask and depth define an observed camera-frame point cloud

$$\mathcal{P}=\left\{z(u,v)K^{-1}[u,v,1]^T\;\middle|\;(u,v)\in M,\;z(u,v)>0\right\}.$$

The task is to estimate the rigid object-to-camera transform

$$T_{co}=\begin{bmatrix}R&\mathbf{t}\\\mathbf{0}^T&1\end{bmatrix}\in SE(3),$$

such that transformed canonical points $R\mathbf{p}_o+\mathbf{t}$ align with the visible portion of $\mathcal{P}$. On labelled data, $(\hat R,\hat{\mathbf{t}})$ is evaluated against ground truth using symmetry-aware rotation error and camera-frame translation error. The test split has no ground-truth poses, so it supports qualitative overlays and registration diagnostics only.

### Coordinate frames and ground-truth transforms

We use three right-handed coordinate frames:

- **Object/canonical frame** $\mathcal{F}_o$: coordinates of the unposed mesh.
- **World frame** $\mathcal{F}_w$: the scene coordinate system used by annotated object poses.
- **Camera frame** $\mathcal{F}_c$: the RGB-D camera coordinate system used by the observed point cloud and final predictions.

Homogeneous transforms act on column vectors. The metadata pose $T_{ow}$ maps canonical object coordinates into world coordinates, while the camera extrinsic $T_{wc}$ maps world coordinates into camera coordinates:

$$\tilde{\mathbf{p}}_w=T_{ow}\tilde{\mathbf{p}}_o, \qquad \tilde{\mathbf{p}}_c=T_{wc}\tilde{\mathbf{p}}_w.$$

Therefore the labelled object-to-camera ground-truth transform used throughout training and evaluation is

$$T_{co}=T_{wc}T_{ow}=\begin{bmatrix}R_{co}&\mathbf{t}_{co}\\\mathbf{0}^T&1\end{bmatrix}.$$

The inverse transforms are $T_{cw}=T_{wc}^{-1}$ and $T_{wo}=T_{ow}^{-1}$. Network input XYZ is centred at the observed-cloud centroid $\bar{\mathbf{p}}_c$, so it predicts $\mathbf{t}_{res}=\mathbf{t}_{co}-\bar{\mathbf{p}}_c$; camera-frame translation is reconstructed as $\hat{\mathbf{t}}_{co}=\bar{\mathbf{p}}_c+\hat{\mathbf{t}}_{res}$.

### Notation

| Symbol | Meaning |
|---|---|
| $I, D$ | RGB image and depth image |
| $K$ | Camera intrinsic matrix |
| $M$ | Instance segmentation mask |
| $o, \mathcal{M}_o$ | Object class and its canonical mesh |
| $\mathcal{F}_o,\mathcal{F}_w,\mathcal{F}_c$ | Canonical object, world, and camera coordinate frames |
| $\mathbf{p}_o,\mathbf{p}_w,\mathbf{p}_c$ | A 3D point expressed in object, world, or camera coordinates |
| $\mathcal{P}$ | Observed camera-frame point cloud extracted from $D$ and $M$ |
| $T_{ab}$ | Homogeneous transform mapping coordinates from frame $b$ into frame $a$ |
| $T_{ow},T_{wc},T_{co}$ | Object-to-world, world-to-camera, and object-to-camera transforms |
| $R,\mathbf{t}$ | Rotation matrix and translation vector of a rigid pose |
| $\hat R,\hat{\mathbf{t}}$ | Predicted rotation and translation; a hat denotes an estimate |
| $\bar{\mathbf{p}}_c,\mathbf{t}_{res}$ | Observed-cloud centroid and centroid-relative translation residual |
| $\mathbf{x}_i$ | Nine-dimensional feature vector of observed point $i$ |
| $\mathbf{e}_o,\mathbf{h}$ | Object-ID embedding and pooled global network feature |
| $\mathbf{a}_1,\mathbf{a}_2$ | Two unconstrained 3-vectors predicted for the 6D rotation representation |
| $\mathcal{S},\mathbf{a}$ | Discrete symmetry group and continuous symmetry axis |
| $\mathbf{g}_i,\mathbf{f}_i,m_i$ | Geometry feature, RGB feature, and valid-image-crop flag for point $i$ |
| $\mathbf{p}_i,\mathbf{q}_i,\mathbf{n}_i$ | Mesh point, matched observed point, and observed-point normal used by ICP |
| $\lambda_{pm},\lambda_{geo},\lambda_t$ | Weights for point-matching, rotation, and translation loss terms |

## 2. Data and Preprocessing

### Dataset overview

The dataset contains cluttered tabletop RGB-D scenes captured with a Kinect-style camera. Each scene provides a color image, aligned depth image, per-pixel instance-label image, camera extrinsic matrix, and metadata identifying the visible objects. Each known object class also has a canonical mesh, physical dimensions, and a scale factor. Labelled train and validation scenes include world-frame object poses; the held-out test scenes omit poses and are used only for qualitative inference. The model is class-conditioned over 79 object categories and learns from visible object instances rather than entire scenes.

Masked depth is back-projected with $K$, voxel-downsampled, and sampled to 1,024 points. Each point contains XYZ, RGB, and camera-oriented normals:

$$[x,y,z,r,g,b,n_x,n_y,n_z].$$

For a depth pixel $(u,v)$ with depth $z$, camera-frame coordinates are recovered by

$$\mathbf{p}_c=zK^{-1}[u,v,1]^T.$$

The object-to-camera pose maps a canonical object point $\mathbf{p}_o$ to

$$\mathbf{p}_c=R\mathbf{p}_o+\mathbf{t}, \qquad R\in SO(3).$$

The cloud is centred and translation is predicted as $t=\bar p+t_{res}$. Separate train, validation, and test caches are used; train supports memory-mapped reads. Training augmentation includes jitter, partial-view cropping, and label-consistent 90°/180° rotations.

## 3. Method

### Architecture

The class-conditioned PointNet applies a shared $9\rightarrow64\rightarrow128\rightarrow1024$ point MLP, max pools to a global feature, appends an object-ID embedding, and predicts 6D rotation plus centroid-relative translation with separate heads. Formally, with point features $\mathbf{x}_i$ and object embedding $\mathbf{e}_o$,

$$\mathbf{h}=\max_i\phi(\mathbf{x}_i), \qquad (\hat{\mathbf{r}}_{6D},\hat{\mathbf{t}}_{res})=f([\mathbf{h},\mathbf{e}_o]).$$

The predicted translation is $\hat{\mathbf{t}}=\bar{\mathbf{p}}+\hat{\mathbf{t}}_{res}$. For the 6D rotation, the two predicted vectors are orthogonalized as

$$\mathbf{b}_1=\frac{\mathbf{a}_1}{\|\mathbf{a}_1\|}, \quad \mathbf{b}_2=\frac{\mathbf{a}_2-(\mathbf{b}_1^T\mathbf{a}_2)\mathbf{b}_1}{\|\mathbf{a}_2-(\mathbf{b}_1^T\mathbf{a}_2)\mathbf{b}_1\|}, \quad \hat R=[\mathbf{b}_1,\mathbf{b}_2,\mathbf{b}_1\times\mathbf{b}_2].$$

The training objective is a weighted combination of point matching, symmetry-aware rotation, and translation terms:

$$\mathcal{L}=\lambda_{pm}\mathcal{L}_{pm}+\lambda_{geo}\mathcal{L}_{geo}+\lambda_t\|\hat{\mathbf{t}}_{res}-\mathbf{t}_{res}\|_1.$$ 

### Rotation representation

Rotations have three physical degrees of freedom but are often represented with more network outputs. A **3D Euler-angle** representation is compact, but it has discontinuities at angle wrapping and singularities (gimbal lock). A **4D quaternion** avoids gimbal lock, but requires unit normalization and has a sign ambiguity: $q$ and $-q$ represent the same rotation. A **9D matrix** representation is easy to interpret, but a network can output an arbitrary non-orthogonal matrix; projecting it back to $SO(3)$ usually requires an SVD or another orthogonalization step.

We use the **6D representation** proposed by two unconstrained 3-vectors $[\mathbf{a}_1,\mathbf{a}_2]$. Gram--Schmidt produces the first two orthonormal matrix columns and their cross product produces the third. This has a continuous mapping over the rotations encountered by the regressor, avoids quaternion sign choices and Euler singularities, and enforces a proper rotation matrix at inference. It is therefore a practical compromise between a minimal parameterization and a redundant 9D matrix output.

RGB–point fusion is a separate train-from-scratch ablation. It tests whether pixel-aligned appearance information can complement geometry, but is not presented as a complete reproduction of DenseFusion.

The object RGB crop is multiplied by its instance mask and augmented with the binary mask as a fourth input channel. A compact CNN converts that crop into a dense feature map $F_I\in\mathbb{R}^{C\times H'\times W'}$. For each observed camera-frame point $\mathbf{p}_i=(x_i,y_i,z_i)$, its image location is

$$u_i=f_x\frac{x_i}{z_i}+c_x, \qquad v_i=f_y\frac{y_i}{z_i}+c_y.$$

The corresponding CNN feature is gathered by bilinear sampling, $\mathbf{f}_i=F_I(u_i,v_i)$. In parallel, the point encoder produces a geometric feature $\mathbf{g}_i=\phi(\mathbf{x}_i)$. A mask-validity flag $m_i$ records whether the projected point lands inside the crop. The fused point representation is

$$\mathbf{h}_i=\psi([\mathbf{g}_i,\mathbf{f}_i,m_i]), \qquad \mathbf{h}=\max_i\mathbf{h}_i,$$

where $\psi$ is a shared 1D fusion block. The pooled fused feature, pooled geometry skip feature, and object embedding are concatenated before the same 6D-rotation and translation heads used by PointNet.

```text
masked RGB crop ──> CNN ──> dense image features ─┐
                                                    ├─> point-wise fusion ─> max pool ─> pose heads
camera-space point cloud ─> Point MLP ─────────────┘
                         └─> project with K ─> bilinear image-feature lookup
```

Synthetic rigid point-cloud rotations are disabled for this ablation because they would break real RGB-to-point correspondences. Partial-view cropping remains enabled because it does not alter the image-to-geometry projection relationship.

### Symmetry handling

The project uses a manually verified symmetry table rather than blindly trusting the CSV field. This matters because the visible geometry of an object such as a can, cup, brick, or bottle may be unchanged under one or more rotations. Penalizing an equivalent orientation would teach the network contradictory targets and would overstate the measured error.

For a discrete symmetry group $\mathcal{S}$, the rotation loss is

$$\mathcal{L}_{geo}(\hat R,R)=\min_{S\in\mathcal{S}}\cos^{-1}\left(\frac{\mathrm{tr}(\hat R^T R S)-1}{2}\right).$$

For a continuous symmetry axis $\mathbf{a}$, it becomes axis-alignment error:

$$\mathcal{L}_{axis}=\cos^{-1}\left((\hat R\mathbf{a})^T(R\mathbf{a})\right).$$

- For discrete symmetries, loss and metrics take the minimum geodesic error over valid target rotations.
- For continuous symmetry, the loss measures symmetry-axis alignment instead of arbitrary spin.
- Symmetric point matching uses closest points, so equivalent surfaces are not penalized.

The minimum above is **not** an unconstrained “best of $N$” prediction rule. The set $\mathcal{S}$ is fixed by the object's known physical symmetry, so every candidate $RS$ is an objectively equivalent ground-truth pose. An arbitrary best-of-$N$ loss would let a model propose unrelated poses and receive credit for whichever happens to be closest; it also requires a separate confidence/ranking mechanism at inference. Our multi-hypothesis experiment tested that direction and did not reliably rank candidates. Using the known symmetry group instead gives a single, well-defined symmetry-invariant target, stable gradients, and a metric that matches the object's geometry.

For continuous symmetry, enumerating a finite number of rotations would introduce an arbitrary angular discretization. Axis alignment removes that artificial resolution choice: any rotation around the object's continuous symmetry axis is treated as equivalent, while tilting the axis is penalized. This is both physically correct and computationally cheaper than searching many sampled rotations.

### Guarded ICP

| Stage | Objective | Threshold |
|---|---|---:|
| 1 | Point-to-plane | 1.0 cm |
| 2 | Point-to-plane | 0.5 cm |
| 3 | Point-to-point | 0.25 cm |

The mesh-to-crop result is accepted only when fitness/RMSE is competitive and it remains within 10° rotation and 1 cm translation of the learned pose. Otherwise PointNet is retained.

The point-to-plane stages minimize the local registration objective

$$E(R,\mathbf{t})=\sum_i\left[\mathbf{n}_i^T(R\mathbf{p}_i+\mathbf{t}-\mathbf{q}_i)\right]^2,$$

where $\mathbf{p}_i$ is a mesh point, $\mathbf{q}_i$ is its matched observed point, and $\mathbf{n}_i$ is the target normal. The acceptance guard requires a competitive registration score and

$$\angle(\hat R^T R_{ICP})\leq10^\circ, \qquad \|\hat{\mathbf{t}}-\mathbf{t}_{ICP}\|\leq0.01\text{ m}.$$

## 4. Experiment

The controlled ablation evaluates 1,705 valid instances from 236 corrected validation scenes. PointNet and RGB–point fusion use the same reset point-sampling seed. Every model is inferred once; pre- and post-ICP metrics come from the same pass.

Metrics are symmetry-aware rotation error, camera-frame translation error, and success at $\leq5°$ / $\leq1$ cm.

## 5. Results

| Method | Mean rot. | Median rot. | Mean trans. | Median trans. | 5° / 1 cm success |
|---|---:|---:|---:|---:|---:|
| PointNet | 3.37° | 2.05° | 0.23 cm | 0.18 cm | 92.73% |
| **PointNet + guarded ICP** | **2.58°** | **1.10°** | **0.18 cm** | **0.14 cm** | **95.31%** |
| RGB–point fusion | 3.31° | 1.25° | 0.25 cm | 0.18 cm | 90.79% |
| RGB–point fusion + guarded ICP | 3.17° | 0.98° | 0.22 cm | 0.15 cm | 91.79% |

Guarded ICP improves PointNet by 0.79° mean rotation error (about 23%), 0.05 cm translation error, and 2.58 percentage points of success. RGB–point fusion slightly lowers mean initializer rotation error, but has lower success and benefits far less from ICP.

**PointNet + guarded ICP is the final method.**

The reproducible numerical table and chart are:

- `outputs/main_ablation_summary.csv`
- `outputs/main_ablation_comparison.png`

## 6. Qualitative ICP Improvements

Dark green is ground truth; orange/red or purple is the learned initializer; bright green is accepted ICP.

### PointNet + guarded ICP

![PointNet ICP improvements](output_images/icp_improvements_pointnet.png)

- `2-51-3`, jenga: 8.64° → 0.50°.
- `2-94-19`, mustard bottle: 7.87° → 1.04°.

### RGB–point fusion + guarded ICP

![Fusion ICP improvements](output_images/icp_improvements_fusion.png)

- `2-63-13`, potted meat can: 8.76° → 0.50°.
- `1-4-10`, potted meat can: 7.59° → 0.32°.

These examples show ICP’s intended role: correct small local errors when the learned pose is already inside the correct registration basin.

## 7. Failure Cases

Large orientation ambiguity under occlusion remains difficult. In `2-6-3`, the side-lying mustard bottle is predicted upright-like by both learned initializers. Local ICP cannot safely solve a roughly 90° error and appropriately rejects unsupported global jumps.

![Side-lying failure across ablations](outputs/ablation_visual_2-6-3_mustard_bottle.png)

Canonical-source soft correspondence, four-hypothesis PointNet, per-point confidence weighting, and hard-example replay fine-tuning did not improve the main table. They are retained as negative ablations; the validated gain comes from reliable local geometric refinement rather than architecture complexity.

## 8. Future Work

1. Collect or synthesize more realistic side-lying and heavily occluded examples.
2. Score a small set of rendered RGB-D pose hypotheses to resolve large discrete orientation ambiguity.
3. Use visibility-aware mesh matching, comparing only surfaces expected to be visible.
4. Benchmark PointNet++ or sparse local-geometry encoders against the established baseline.
5. Predict pose uncertainty so ambiguous objects can request another view.

## 9. Reproducibility

```bash
python pose_pipeline.py ablations
python pose_pipeline.py visualize-failure --scene 2-6-3 --object mustard_bottle
python pose_pipeline.py visualize-icp --model pointnet
python pose_pipeline.py visualize-icp --model fusion
```

Primary checkpoint: `model_weights/pointnet_occlusion_guarded_v1.pth`.
