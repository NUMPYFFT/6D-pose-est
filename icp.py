import os
import numpy as np
import trimesh
import open3d as o3d
import threading
import concurrent.futures
import hashlib
from pathlib import Path
from tqdm import tqdm

# Global cache for canonical models to avoid reloading in workers
_canonical_model_cache = {}
_cache_lock = threading.Lock()
_PROJECT_ROOT = Path(__file__).resolve().parent
_CANONICAL_CACHE_DIR = _PROJECT_ROOT / "models" / "canonical_point_cache"


def _canonical_cache_path(obj_name):
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(obj_name))
    return _CANONICAL_CACHE_DIR / f"{safe_name}.npz"


def _mesh_path_from_row(row):
    location = Path(str(row["location"]))
    if not location.is_absolute():
        location = _PROJECT_ROOT / location
    return location / "visual_meshes" / "visual.dae"


def _sample_mesh_deterministically(obj_name, row, num_points=10000):
    """Sample a reproducible canonical cloud directly from one object mesh."""
    mesh_path = _mesh_path_from_row(row)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Canonical mesh is missing: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")
    seed = int.from_bytes(hashlib.sha256(str(obj_name).encode("utf-8")).digest()[:4], "little")
    points, _ = trimesh.sample.sample_surface(mesh, num_points, seed=seed)
    return points.astype(np.float32)


def build_canonical_point_cache(objects_df, num_points=10000, overwrite=False):
    """Build one deterministic, unscaled mesh-surface cloud per catalog object."""
    _CANONICAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    built = skipped = failed = 0
    for _, row in tqdm(
        objects_df.iterrows(), total=len(objects_df), desc="Canonical mesh cache", unit="object"
    ):
        obj_name = str(row["object"])
        cache_path = _canonical_cache_path(obj_name)
        if cache_path.is_file() and not overwrite:
            skipped += 1
            continue
        try:
            points = _sample_mesh_deterministically(obj_name, row, num_points=num_points)
            np.savez_compressed(cache_path, points=points)
            built += 1
        except Exception as exc:
            failed += 1
            print(f"[WARN] Could not cache {obj_name}: {exc}")
    print(f"Canonical cache ready: {built} built, {skipped} existing, {failed} failed.")
    return {"built": built, "skipped": skipped, "failed": failed}


def _load_cached_canonical_points(obj_name, row, num_points=10000):
    cache_path = _canonical_cache_path(obj_name)
    with _cache_lock:
        if not cache_path.is_file():
            _CANONICAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            points = _sample_mesh_deterministically(obj_name, row, num_points=num_points)
            np.savez_compressed(cache_path, points=points)
        with np.load(cache_path) as cached:
            return cached["points"].astype(np.float32, copy=True)

# -------------------------------------------------------------------------
# Load canonical model using CSV['location']
# -------------------------------------------------------------------------
def load_canonical_model_from_csv(obj_name, objects_df, scale):
    scale_array = None
    if scale is not None:
        scale_array = np.asarray(scale, dtype=np.float32).reshape(-1)
        if scale_array.size == 1:
            scale_array = np.repeat(scale_array, 3)
    # Check cache first
    cache_key = (obj_name, tuple(scale_array) if scale_array is not None else None)
    with _cache_lock:
        if cache_key in _canonical_model_cache:
            return _canonical_model_cache[cache_key]

    row = objects_df[objects_df['object'] == obj_name]
    if row.empty:
        print(f"[WARN] Object {obj_name} not found in CSV.")
        return None

    try:
        pts = _load_cached_canonical_points(obj_name, row.iloc[0])
    except Exception as e:
        print(f"[ERROR] Failed to load canonical points for {obj_name}: {e}")
        return None

    # Apply scale
    if scale_array is not None:
        pts = pts * scale_array[None, :]

    # Convert to Open3D PCD
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(0.005)  # 5mm
    pcd.estimate_normals()

    with _cache_lock:
        _canonical_model_cache[cache_key] = pcd
    return pcd


# -------------------------------------------------------------------------
# Pure-ICP pose estimation (no learned prior): global registration + local ICP.
# Used to validate that classical registration alone can hit the 5deg/1cm target
# before any PointNet prediction is involved.
# -------------------------------------------------------------------------
def _preprocess_for_global_registration(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_down, fpfh


def global_registration(source_pcd, target_pcd, voxel_size=0.01):
    """RANSAC + FPFH feature matching, including the retained correspondence count."""
    source_down, source_fpfh = _preprocess_for_global_registration(source_pcd, voxel_size)
    target_down, target_fpfh = _preprocess_for_global_registration(target_pcd, voxel_size)

    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000))
    return result.transformation, result.fitness, len(result.correspondence_set)


def refine_icp_unconstrained(init_pose, source_pcd, target_pcd, max_iter=50, stages=3, coarse_threshold=0.02):
    """Same 3-stage point-to-plane/point-to-point polish as refine_icp(), but with no
    trusted-prediction safety check -- here ICP's own output IS the final pose, there is
    no separate PointNet estimate to fall back on."""
    current_pose = init_pose
    reg = None
    try:
        reg = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, coarse_threshold, current_pose,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
        if reg.fitness > 0:
            current_pose = reg.transformation

        if stages >= 2:
            reg = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, coarse_threshold * 0.5, current_pose,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
            if reg.fitness > 0:
                current_pose = reg.transformation

        if stages >= 3:
            reg = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, coarse_threshold * 0.25, current_pose,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20))
            if reg.fitness > 0:
                current_pose = reg.transformation
    except Exception as e:
        print(f"[ICP FAIL] {e}")
        return init_pose[:3, :3], init_pose[:3, 3], 0.0, 0.0

    fitness = reg.fitness if reg is not None else 0.0
    rmse = reg.inlier_rmse if reg is not None else 0.0
    return current_pose[:3, :3], current_pose[:3, 3], fitness, rmse


def _canonical_rotation_grid():
    """24 axis-aligned rotation candidates (the cube's rotation group)."""
    rotations = []
    for x_deg in [0, 90, 180, 270]:
        for y_deg in [0, 90, 180, 270]:
            for z_deg in [0, 90, 180, 270]:
                rx, ry, rz = np.radians([x_deg, y_deg, z_deg])
                Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
                Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
                Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
                R = Rz @ Ry @ Rx
                if not any(np.allclose(R, r, atol=1e-3) for r in rotations):
                    rotations.append(R)
    return rotations


def _best_centroid_aligned_pose(source_pts, target_pts):
    """Cheap fallback initial pose: centroid alignment + brute-force search over 24 canonical
    rotations, scored by mean nearest-neighbor distance. Used when RANSAC finds too few
    FPFH correspondences (common on small/symmetric/low-texture-geometry objects)."""
    from sklearn.neighbors import NearestNeighbors
    src_centroid = source_pts.mean(axis=0)
    tgt_centroid = target_pts.mean(axis=0)
    sample = target_pts if len(target_pts) <= 2000 else target_pts[
        np.random.choice(len(target_pts), 2000, replace=False)]
    nbrs = NearestNeighbors(n_neighbors=1).fit(sample)

    best_dist, best_pose = np.inf, np.eye(4)
    for R_cand in _canonical_rotation_grid():
        t_cand = tgt_centroid - R_cand @ src_centroid
        transformed = (R_cand @ source_pts.T).T + t_cand
        dists, _ = nbrs.kneighbors(transformed)
        mean_d = dists.mean()
        if mean_d < best_dist:
            best_dist = mean_d
            best_pose = np.eye(4)
            best_pose[:3, :3] = R_cand
            best_pose[:3, 3] = t_cand
    return best_pose


def estimate_pose_pure_icp(obj_name, objects_df, scale, obs_pts, voxel_size=0.01, icp_stages=3,
                            icp_threshold=0.02, ransac_fitness_floor=0.2,
                            min_ransac_correspondences=15):
    """Full pose estimate from geometry alone: global registration (FPFH+RANSAC) for the
    initial guess, then local ICP refinement -- no PointNet prediction involved. Falls back
    to a centroid + canonical-rotation search when RANSAC's fitness is too low to trust
    (its FPFH correspondences frequently collapse to single digits on small/symmetric objects)."""
    source_pcd = load_canonical_model_from_csv(obj_name, objects_df, scale)
    if source_pcd is None or len(obs_pts) <= 30:
        return np.eye(3), np.zeros(3), 0.0, 0.0

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(obs_pts)
    target_pcd = target_pcd.voxel_down_sample(0.005)
    target_pcd.estimate_normals()
    target_pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))

    if len(target_pcd.points) <= 30:
        return np.eye(3), np.zeros(3), 0.0, 0.0

    init_pose, ransac_fitness, correspondence_count = global_registration(
        source_pcd, target_pcd, voxel_size=voxel_size)

    # Do not polish an underconstrained RANSAC result: sparse FPFH matches can yield a
    # numerically valid transform that is geometrically meaningless. Start ICP from the
    # centroid-aligned rotation-grid search instead.
    used_fallback = correspondence_count < min_ransac_correspondences
    if used_fallback:
        init_pose = _best_centroid_aligned_pose(
            np.asarray(source_pcd.points), np.asarray(target_pcd.points))

    R, t, fitness, rmse = refine_icp_unconstrained(
        init_pose, source_pcd, target_pcd, stages=icp_stages, coarse_threshold=icp_threshold)

    if not used_fallback and (ransac_fitness < ransac_fitness_floor or fitness < ransac_fitness_floor):
        fallback_pose = _best_centroid_aligned_pose(
            np.asarray(source_pcd.points), np.asarray(target_pcd.points))
        R2, t2, fitness2, rmse2 = refine_icp_unconstrained(
            fallback_pose, source_pcd, target_pcd, stages=icp_stages, coarse_threshold=icp_threshold)
        if fitness2 > fitness:
            R, t, fitness, rmse = R2, t2, fitness2, rmse2

    return R, t, fitness, rmse


def process_pure_icp_sample(data):
    """Worker function: pure-ICP pose estimation for a single sample (no pred_R/pred_t input)."""
    R, t, fitness, rmse = estimate_pose_pure_icp(
        data['obj_name'], data['objects_df'], data['scale'], data['obs_pts'],
        voxel_size=data.get('voxel_size', 0.01),
        icp_stages=data['icp_stages'], icp_threshold=data['icp_threshold'])
    return R, t, fitness, rmse


def run_pure_icp(samples_to_process):
    """Fill in 'pred_R'/'pred_t' for every sample using pure-ICP (global registration + ICP),
    completely independent of any PointNet prediction."""
    print(f"Running pure ICP (no PointNet) on {len(samples_to_process)} samples...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_pure_icp_sample, s): i for i, s in enumerate(samples_to_process)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(samples_to_process), desc="Pure ICP"):
            i = futures[future]
            try:
                R, t, fitness, rmse = future.result()
                samples_to_process[i]['pred_R'] = R
                samples_to_process[i]['pred_t'] = t
                samples_to_process[i]['icp_fitness'] = fitness
                samples_to_process[i]['icp_rmse'] = rmse
            except Exception as e:
                print(f"Pure ICP worker failed for sample {i}: {e}")
                samples_to_process[i]['pred_R'] = np.eye(3)
                samples_to_process[i]['pred_t'] = np.zeros(3)
                samples_to_process[i]['icp_fitness'] = 0.0
                samples_to_process[i]['icp_rmse'] = 0.0


# -------------------------------------------------------------------------
# ICP refinement (Robust 3-Stage)
# -------------------------------------------------------------------------
def refine_icp(pred_R, pred_t, source_pcd, target_pcd, max_iter=50, stages=3,
               coarse_threshold=0.02, max_rot_change=30.0,
               max_translation_change=0.01):
    # Initial Pose from PointNet
    T_init = np.eye(4, dtype=np.float32)
    T_init[:3, :3] = pred_R
    T_init[:3, 3] = pred_t

    current_pose = T_init

    try:
        # Score the learned pose before refinement.  ICP is only allowed to
        # replace it if the final alignment is not worse at the same scale.
        # This is important for partial or symmetric observations, where ICP
        # can otherwise drift into a locally plausible but incorrect pose.
        eval_threshold = coarse_threshold * 0.5
        init_eval = o3d.pipelines.registration.evaluate_registration(
            source_pcd, target_pcd, eval_threshold, T_init)

        # Stage 1: Coarse alignment (default 2cm threshold) - Point-to-Plane
        reg_p2l_coarse = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, coarse_threshold, current_pose,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )
        
        if reg_p2l_coarse.fitness > 0:
            current_pose = reg_p2l_coarse.transformation

        # Stage 2: Fine alignment (1cm threshold) - Point-to-Plane
        if stages >= 2:
            reg_p2l_fine = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, coarse_threshold * 0.5, current_pose,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            
            if reg_p2l_fine.fitness > 0:
                current_pose = reg_p2l_fine.transformation

        # Stage 3: Fine alignment (0.5cm threshold) - Point-to-Point (Fix sliding)
        if stages >= 3:
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, coarse_threshold * 0.25, current_pose,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
            )
            
            # Accept if RMSE improved or Fitness improved
            # We prioritize RMSE for Point-to-Point as it tightens the fit
            if reg_p2p.fitness > 0:
                 current_pose = reg_p2p.transformation

        # -------------------------------------------------------
        # SAFETY CHECK: Rotation & Translation Deviation
        # -------------------------------------------------------
        R_final = current_pose[:3, :3]
        t_final = current_pose[:3, 3]
        
        # Rotation Diff
        R_diff = R_final @ pred_R.T
        trace = np.trace(R_diff)
        cos_theta = (trace - 1) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_diff = np.degrees(np.arccos(cos_theta))

        # Translation Diff
        trans_diff = np.linalg.norm(t_final - pred_t)

        if angle_diff > max_rot_change:
            # print(f"[ICP REJECT] Rotation change {angle_diff:.2f} > {max_rot_change} deg.")
            return pred_R, pred_t, 0.0, 0.0
            
        if trans_diff > max_translation_change:
            # print(f"[ICP REJECT] Translation change {trans_diff*100:.2f} > 1.0 cm.")
            return pred_R, pred_t, 0.0, 0.0

        final_eval = o3d.pipelines.registration.evaluate_registration(
            source_pcd, target_pcd, eval_threshold, current_pose)
        fit_not_worse = final_eval.fitness >= init_eval.fitness - 1e-3
        rmse_not_worse = (
            init_eval.fitness == 0 or
            final_eval.inlier_rmse <= init_eval.inlier_rmse * 1.02
        )
        meaningful_gain = (
            final_eval.fitness > init_eval.fitness + 1e-3 or
            (init_eval.fitness > 0 and
             final_eval.inlier_rmse < init_eval.inlier_rmse * 0.98)
        )
        if not (fit_not_worse and rmse_not_worse and meaningful_gain):
            return pred_R, pred_t, 0.0, 0.0

    except Exception as e:
        print(f"[ICP FAIL] {e}")
        return pred_R, pred_t, 0.0, 0.0

    # Get final metrics from the last successful registration
    final_fitness = 0.0
    final_rmse = 0.0
    if stages >= 3 and 'reg_p2p' in locals():
        final_fitness = reg_p2p.fitness
        final_rmse = reg_p2p.inlier_rmse
    elif stages >= 2 and 'reg_p2l_fine' in locals():
        final_fitness = reg_p2l_fine.fitness
        final_rmse = reg_p2l_fine.inlier_rmse
    elif 'reg_p2l_coarse' in locals():
        final_fitness = reg_p2l_coarse.fitness
        final_rmse = reg_p2l_coarse.inlier_rmse

    return current_pose[:3, :3], current_pose[:3, 3], final_fitness, final_rmse


def _large_turn_candidate_specs(pred_R, angles_degrees=(-90, 90)):
    """Return labelled large horizontal-orientation alternatives to a PointNet pose.

    Axis-aligned and diagonal horizontal axes cover objects lying in different
    tabletop bearings without turning this fallback into an unrestricted search.
    """
    candidates = []
    axes = (
        ("x", np.array([1.0, 0.0, 0.0])),
        ("y", np.array([0.0, 1.0, 0.0])),
        ("x+y", np.array([1.0, 1.0, 0.0]) / np.sqrt(2)),
        ("x-y", np.array([1.0, -1.0, 0.0]) / np.sqrt(2)),
    )
    for axis_name, axis in axes:
        for angle_degrees in angles_degrees:
            angle = np.radians(angle_degrees)
            c, s = np.cos(angle), np.sin(angle)
            skew = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
            rotation = np.eye(3) + s * skew + (1 - c) * (skew @ skew)
            turn = f"{angle_degrees:+g}"
            candidates.append((f"{turn} deg about {axis_name}", rotation @ pred_R))
    return candidates


def _quarter_turn_candidate_specs(pred_R):
    """Return the production fallback's eight ±90° rotation proposals."""
    return _large_turn_candidate_specs(pred_R, angles_degrees=(-90, 90))


def _quarter_turn_candidates(pred_R):
    """Compatibility wrapper that returns only candidate rotation matrices."""
    return [rotation for _, rotation in _quarter_turn_candidate_specs(pred_R)]


def _pose_alignment_score(rotation, translation, source_pcd, target_pcd, threshold):
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return o3d.pipelines.registration.evaluate_registration(
        source_pcd, target_pcd, threshold, pose
    )


def _rotation_error_degrees(rotation, reference_rotation):
    """Geodesic rotation distance for diagnostic reporting only."""
    trace = np.trace(rotation.T @ reference_rotation)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def refine_icp_quarter_turn_fallback(pred_R, pred_t, source_pcd, target_pcd, stages,
                                     coarse_threshold, max_rot_change,
                                     min_fitness=0.15, min_fitness_gain=0.05):
    """Try eight ±90° horizontal hypotheses only when normal ICP has no support.

    Each candidate is still locally guarded by :func:`refine_icp`; selection then
    requires a clear geometric improvement over the original PointNet pose.  This
    permits recovery from upright-versus-sideways ambiguity without applying an
    expensive or risky global search to ordinary well-aligned objects.
    """
    score_threshold = coarse_threshold * 0.5
    base_score = _pose_alignment_score(
        pred_R, pred_t, source_pcd, target_pcd, score_threshold
    )
    best = None
    for candidate_R in _quarter_turn_candidates(pred_R):
        candidate_R, candidate_t, _, _ = refine_icp(
            candidate_R, pred_t, source_pcd, target_pcd,
            stages=stages, coarse_threshold=coarse_threshold,
            # A 90° proposal can still be 10--20° from the true pose.  This
            # wider limit applies only after proposing a discrete large turn;
            # final acceptance is still based on observed geometry.
            max_rot_change=max(max_rot_change, 20.0),
            # The coarse PointNet translation can be a few centimetres off on
            # the same cases that need a quarter-turn. Final selection remains
            # geometry-gated, so only this fallback gets the wider local window.
            max_translation_change=0.03,
        )
        score = _pose_alignment_score(
            candidate_R, candidate_t, source_pcd, target_pcd, score_threshold
        )
        if best is None or (score.fitness, -score.inlier_rmse) > (best[2].fitness, -best[2].inlier_rmse):
            best = (candidate_R, candidate_t, score)

    if best is None:
        return pred_R, pred_t, 0.0, 0.0, False

    best_R, best_t, best_score = best
    has_enough_support = best_score.fitness >= max(min_fitness, base_score.fitness + min_fitness_gain)
    rmse_competitive = (
        base_score.fitness == 0
        or best_score.inlier_rmse <= base_score.inlier_rmse * 1.05
    )
    if not (has_enough_support and rmse_competitive):
        return pred_R, pred_t, 0.0, 0.0, False
    return best_R, best_t, best_score.fitness, best_score.inlier_rmse, True


def diagnose_quarter_turn_candidates(pred_R, pred_t, source_pcd, target_pcd, stages,
                                     coarse_threshold, max_rot_change,
                                     min_fitness=0.15, min_fitness_gain=0.05,
                                     gt_R=None, gt_t=None,
                                     candidate_angles=(-90, 90)):
    """Return transparent scoring details for each large-turn ICP hypothesis.

    This is intentionally diagnostic-only: it does not modify a prediction or
    relax any production acceptance rule.  It is useful for a single failed
    object when deciding whether more hypotheses would be justified.
    """
    score_threshold = coarse_threshold * 0.5
    base_score = _pose_alignment_score(
        pred_R, pred_t, source_pcd, target_pcd, score_threshold
    )
    required_fitness = max(min_fitness, base_score.fitness + min_fitness_gain)
    rows = []
    for label, initial_R in _large_turn_candidate_specs(
        pred_R, angles_degrees=candidate_angles
    ):
        before = _pose_alignment_score(
            initial_R, pred_t, source_pcd, target_pcd, score_threshold
        )
        refined_R, refined_t, local_fitness, local_rmse = refine_icp(
            initial_R, pred_t, source_pcd, target_pcd,
            stages=stages, coarse_threshold=coarse_threshold,
            max_rot_change=max(max_rot_change, 20.0),
            max_translation_change=0.03,
        )
        after = _pose_alignment_score(
            refined_R, refined_t, source_pcd, target_pcd, score_threshold
        )
        rmse_competitive = (
            base_score.fitness == 0
            or after.inlier_rmse <= base_score.inlier_rmse * 1.05
        )
        row = {
            "candidate": label,
            "before_fitness": float(before.fitness),
            "before_rmse": float(before.inlier_rmse),
            "local_icp_accepted": bool(local_fitness > 0.0),
            "after_fitness": float(after.fitness),
            "after_rmse": float(after.inlier_rmse),
            "meets_support_gate": bool(after.fitness >= required_fitness),
            "meets_rmse_gate": bool(rmse_competitive),
            "would_be_accepted": bool(after.fitness >= required_fitness and rmse_competitive),
        }
        if gt_R is not None:
            row["rotation_error_vs_gt_deg"] = _rotation_error_degrees(refined_R, gt_R)
        if gt_t is not None:
            row["translation_error_vs_gt_cm"] = float(np.linalg.norm(refined_t - gt_t) * 100)
        rows.append(row)
    result = {
        "base_fitness": float(base_score.fitness),
        "base_rmse": float(base_score.inlier_rmse),
        "required_fitness": float(required_fitness),
        "score_threshold": float(score_threshold),
        "candidates": rows,
    }
    if gt_R is not None and gt_t is not None:
        gt_score = _pose_alignment_score(gt_R, gt_t, source_pcd, target_pcd, score_threshold)
        result.update({
            "pointnet_rotation_error_vs_gt_deg": _rotation_error_degrees(pred_R, gt_R),
            "pointnet_translation_error_vs_gt_cm": float(np.linalg.norm(pred_t - gt_t) * 100),
            "ground_truth_fitness": float(gt_score.fitness),
            "ground_truth_rmse": float(gt_score.inlier_rmse),
        })
    return result


def diagnose_icp_sample(data):
    """Build one sample's point clouds and return quarter-turn score diagnostics."""
    source_pcd = load_canonical_model_from_csv(
        data["obj_name"], data["objects_df"], data["scale"]
    )
    if source_pcd is None:
        raise ValueError(f"Could not load canonical model for {data['obj_name']!r}.")
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(data["obs_pts"])
    target_pcd = target_pcd.voxel_down_sample(0.005)
    target_pcd.estimate_normals()
    target_pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
    if len(target_pcd.points) <= 30:
        raise ValueError("The observed crop has too few points for ICP diagnosis.")
    result = diagnose_quarter_turn_candidates(
        data["pred_R"], data["pred_t"], source_pcd, target_pcd,
        stages=data["icp_stages"], coarse_threshold=data["icp_threshold"],
        max_rot_change=data["max_rot_change"],
        gt_R=data.get("gt_R"), gt_t=data.get("gt_t"),
    )
    result.update({
        "object": data["obj_name"],
        "observed_points": int(len(data["obs_pts"])),
        "downsampled_points": int(len(target_pcd.points)),
    })
    return result

# -------------------------------------------------------------------------
# Worker function for Parallel ICP
# -------------------------------------------------------------------------
def process_icp_sample(data):
    """
    Worker function to run ICP on a single sample.
    data: dict containing all necessary inputs
    """
    pred_R = data['pred_R']
    pred_t = data['pred_t']
    # Keep the registration cloud separate from the full visualization cloud.
    obs_pts = data.get('icp_obs_pts', data['obs_pts'])
    obj_name = data['obj_name']
    scale = data['scale']
    icp_stages = data['icp_stages']
    icp_threshold = data['icp_threshold']
    max_rot_change = data['max_rot_change']
    use_quarter_turn_fallback = data.get('use_quarter_turn_fallback', False)
    objects_df = data['objects_df'] # Passed as DF, but could be optimized
    
    # Load canonical model (cached inside worker process)
    source_pcd = load_canonical_model_from_csv(obj_name, objects_df, scale)
    
    if source_pcd is None:
        return pred_R, pred_t, 0.0, 0.0

    # Build target PCD
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(obs_pts)
    target_pcd = target_pcd.voxel_down_sample(0.005)
    target_pcd.estimate_normals()
    target_pcd.orient_normals_towards_camera_location(np.array([0.,0.,0.]))
    
    if len(target_pcd.points) <= 30:
        return pred_R, pred_t, 0.0, 0.0

    # First run the inexpensive trusted-pose refinement for every object.
    refined_R, refined_t, fitness, rmse = refine_icp(pred_R, pred_t, source_pcd, target_pcd, 
                                      stages=icp_stages, 
                                      coarse_threshold=icp_threshold, 
                                      max_rot_change=max_rot_change)
    used_quarter_turn_fallback = False

    # Only rejected/unsupported normal ICP gets the four larger hypotheses.
    # Very sparse crops do not contain enough evidence to distinguish orientations.
    if use_quarter_turn_fallback and fitness == 0.0 and len(obs_pts) >= 200:
        refined_R, refined_t, fitness, rmse, used_quarter_turn_fallback = \
            refine_icp_quarter_turn_fallback(
                pred_R, pred_t, source_pcd, target_pcd,
                stages=icp_stages, coarse_threshold=icp_threshold,
                max_rot_change=max_rot_change,
            )

    return refined_R, refined_t, fitness, rmse, used_quarter_turn_fallback

def run_icp_refinement(samples_to_process, use_icp):
    if use_icp:
        print(f"Running ICP on {len(samples_to_process)} samples using ThreadPoolExecutor...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks
            futures = {executor.submit(process_icp_sample, s): i for i, s in enumerate(samples_to_process)}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(samples_to_process), desc="ICP Refinement"):
                i = futures[future]
                try:
                    refined_R, refined_t, fitness, rmse, used_quarter_turn_fallback = future.result()
                    
                    # Save initial pose before overwriting
                    samples_to_process[i]['init_R'] = samples_to_process[i]['pred_R'].copy()
                    samples_to_process[i]['init_t'] = samples_to_process[i]['pred_t'].copy()
                    
                    samples_to_process[i]['pred_R'] = refined_R
                    samples_to_process[i]['pred_t'] = refined_t
                    samples_to_process[i]['icp_fitness'] = fitness
                    samples_to_process[i]['icp_rmse'] = rmse
                    samples_to_process[i]['icp_quarter_turn_used'] = used_quarter_turn_fallback
                except Exception as e:
                    print(f"ICP Worker failed for sample {i}: {e}")
                    samples_to_process[i]['icp_fitness'] = 0.0
                    samples_to_process[i]['icp_rmse'] = 0.0
                    samples_to_process[i]['icp_quarter_turn_used'] = False
    else:
        # Initialize default values if ICP is off
        for s in samples_to_process:
            s['icp_fitness'] = 0.0
            s['icp_rmse'] = 0.0
            s['icp_quarter_turn_used'] = False
