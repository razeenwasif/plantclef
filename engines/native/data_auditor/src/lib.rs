//! # PlantCLEF 2026 Data Auditor
//!
//! A high-performance systems-level utility for large-scale botanical datasets.
//! Utilizes the Rust/Python duo via PyO3 to bypass the Python GIL and provide 
//! near-instant I/O operations for 1.4M+ image collections.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs;
use std::path::Path;
use std::num::NonZeroU32;
use std::collections::{HashSet, VecDeque};
use indicatif::{ProgressBar, ProgressStyle};
use numpy::{PyArray1, PyArray2, ToPyArray, PyArrayMethods};
use bitvec::prelude::*;
use jwalk::WalkDir;
use polars::prelude::*;
use memmap2::Mmap;
use std::fs::File;
use fast_image_resize as fr;
use turbojpeg::{Decompressor, Subsamp};
use numpy::ndarray::Array2;

/// High-Performance Memory-Mapped Feature Store.
/// Bypasses Python I/O by providing zero-copy access to a unified binary cache.
#[pyclass]
pub struct FeatureStore {
    mmap: Mmap,
    num_samples: usize,
    feat_dim: usize,
}

#[pymethods]
impl FeatureStore {
    #[new]
    pub fn new(path: String, num_samples: usize, feat_dim: usize) -> PyResult<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(FeatureStore { mmap, num_samples, feat_dim })
    }

    /// Retrieve a single feature vector as a numpy array.
    /// Access is O(1) and happens at RAM speed.
    pub fn get_feature<'py>(&self, py: Python<'py>, idx: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
        if idx >= self.num_samples {
            return Err(pyo3::exceptions::PyIndexError::new_err("Sample index out of bounds"));
        }

        // Each float32 is 4 bytes. 
        let start = idx * self.feat_dim * 4;
        let end = start + (self.feat_dim * 4);
        
        let slice = &self.mmap[start..end];
        let f32_slice: &[f32] = unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const f32, self.feat_dim)
        };

        Ok(f32_slice.to_pyarray_bound(py))
    }

    /// ORACLE: Zen 4 Hardware Prefetcher.
    /// Manually warms the L1 cache for a future feature vector.
    pub fn prefetch(&self, idx: usize) -> PyResult<()> {
        if idx < self.num_samples {
            let start = idx * self.feat_dim * 4;
            let ptr = self.mmap.as_ptr().wrapping_add(start);
            unsafe {
                // Prefetch first 4 cache lines (256 bytes) using inline assembly
                // This guarantees the instruction boundary the compiler cannot move.
                core::arch::asm!(
                    "prefetcht0 [{ptr} + 0]",
                    "prefetcht0 [{ptr} + 64]",
                    "prefetcht0 [{ptr} + 128]",
                    "prefetcht0 [{ptr} + 192]",
                    ptr = in(reg) ptr,
                );
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.num_samples
    }
}

/// High-Performance branchless masking using AVX-512.
/// Directly implements the zero-masked move idiom via intrinsics.
#[pyfunction]
fn apply_species_mask_avx512(probs: Vec<f32>, threshold: f32) -> PyResult<Vec<f32>> {
    let n = probs.len();
    let mut output = vec![0.0f32; n];
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            use std::arch::x86_64::*;
            unsafe {
                let mut i = 0;
                let thresh_v = _mm512_set1_ps(threshold);
                while i + 16 <= n {
                    let ptr_in = probs.as_ptr().add(i);
                    let ptr_out = output.as_mut_ptr().add(i);
                    
                    // Load 16 floats
                    let data = _mm512_loadu_ps(ptr_in);
                    // Compare: data >= threshold
                    let mask = _mm512_cmp_ps_mask(data, thresh_v, _CMP_GE_OQ);
                    // Masked store: zeros out lanes where mask is 0
                    _mm512_mask_storeu_ps(ptr_out, mask, data);
                    
                    i += 16;
                }
                // Handle remainder
                for j in i..n {
                    if probs[j] >= threshold { output[j] = probs[j]; }
                }
            }
            return Ok(output);
        }
    }

    // Fallback if hardware doesn't support it
    for i in 0..n {
        if probs[i] >= threshold { output[i] = probs[i]; }
    }
    Ok(output)
}

/// Audit the compression 'DNA' of a sample image set.
#[pyfunction]
fn audit_compression_dna(paths: Vec<String>) -> PyResult<(String, i32)> {
    let mut decompressor = Decompressor::new().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    
    let mut subsamplings = Vec::new();
    
    for path in paths.iter().take(50) {
        let jpeg_data = std::fs::read(path)?;
        let header = decompressor.read_header(&jpeg_data).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        subsamplings.push(header.subsamp);
    }
    
    let mut counts = std::collections::HashMap::new();
    for s in subsamplings {
        *counts.entry(s).or_insert(0) += 1;
    }
    
    // Default to Sub2x2 (4:2:0)
    let most_common = counts.into_iter().max_by_key(|&(_, count)| count).map(|(s, _)| s).unwrap_or(Subsamp::Sub2x2);
    
    let sub_str = format!("{:?}", most_common);
    Ok((sub_str, 95))
}

/// High-Performance SIMD-Accelerated Lanczos3 Resizer.
#[pyfunction]
fn resize_image_lanczos(image_data: Vec<u8>, target_w: u32, target_h: u32) -> PyResult<Vec<u8>> {
    let img = image::load_from_memory(&image_data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .to_rgba8();
        
    let width = NonZeroU32::new(img.width()).unwrap();
    let height = NonZeroU32::new(img.height()).unwrap();
    
    let src_image = fr::images::Image::from_vec_u8(
        width.get(),
        height.get(),
        img.into_raw(),
        fr::PixelType::U8x4,
    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let mut dst_image = fr::images::Image::new(
        target_w,
        target_h,
        fr::PixelType::U8x4,
    );

    let mut resizer = fr::Resizer::new();
    resizer.resize(&src_image, &mut dst_image, None).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let rgba = dst_image.buffer();
    let mut rgb = Vec::with_capacity((target_w * target_h * 3) as usize);
    for chunk in rgba.chunks_exact(4) {
        rgb.push(chunk[0]);
        rgb.push(chunk[1]);
        rgb.push(chunk[2]);
    }
    
    Ok(rgb)
}

/// Parallel Per-Class Threshold Optimization.
/// Searches for the probability cutoff that maximizes F1 for each species.
#[pyfunction]
fn parallel_threshold_search(
    logits: Vec<Vec<f32>>, // [N_samples, C_classes]
    labels: Vec<usize>,    // [N_samples]
    num_classes: usize,
) -> PyResult<Vec<f32>> {
    let n_samples = logits.len();
    
    // Transpose logits to [C, N] for easier parallelization across classes
    let mut class_probs: Vec<Vec<f32>> = vec![Vec::with_capacity(n_samples); num_classes];
    for row in logits {
        for (c, &p) in row.iter().enumerate() {
            if c < num_classes { class_probs[c].push(p); }
        }
    }

    // Search candidates
    let candidates: Vec<f32> = (1..100).map(|i| i as f32 / 200.0).collect(); // 0.005 to 0.5

    let best_thresholds: Vec<f32> = class_probs.into_par_iter().enumerate().map(|(c, probs)| {
        let mut best_f1 = -1.0;
        let mut best_t = 0.1;
        
        for &t in &candidates {
            let mut tp = 0.0;
            let mut fp = 0.0;
            let mut fn_count = 0.0;
            
            for (i, &p) in probs.iter().enumerate() {
                // ORACLE OPTIMIZATION: Branchless SIMD-friendly accumulators
                let is_true = (labels[i] == c) as u32 as f32;
                let is_pred = (p > t) as u32 as f32;
                
                tp += is_pred * is_true;
                fp += is_pred * (1.0 - is_true);
                fn_count += (1.0 - is_pred) * is_true;
            }
            
            let f1 = if (2.0 * tp + fp + fn_count) > 0.0 {
                (2.0 * tp) / (2.0 * tp + fp + fn_count)
            } else { 0.0 };
            
            if f1 > best_f1 {
                best_f1 = f1;
                best_t = t;
            }
        }
        best_t
    }).collect();

    Ok(best_thresholds)
}

/// High-Performance Weighted Metadata Sampler.
#[pyfunction]
fn sample_epoch_indices(
    counts: Vec<usize>, 
    labels: Vec<usize>,
    strategy: String,
    seed: u64
) -> PyResult<Vec<usize>> {
    use rand::prelude::*;
    use rand::distributions::WeightedIndex;
    
    let mut rng = StdRng::seed_from_u64(seed);
    let num_samples = labels.len();
    
    // 1. Calculate weights based on strategy
    let mut weights = Vec::with_capacity(num_samples);
    for &l in &labels {
        let count = counts[l] as f64;
        let w = match strategy.as_str() {
            "sqrt" => 1.0 / count.sqrt(),
            "balanced" => 1.0 / count,
            _ => 1.0, // Natural
        };
        weights.push(w);
    }
    
    // 2. Weighted sampling
    let dist = WeightedIndex::new(&weights).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let mut indices = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        indices.push(dist.sample(&mut rng));
    }
    
    Ok(indices)
}
///
/// Bypasses Python's slow CSV parsing by using Polars' native multi-threaded
/// IPC/CSV readers. Memory-maps feather files for zero-copy performance.
///
/// # Arguments
/// * `file_path` - Path to the .csv or .feather metadata file.
///
/// # Returns
/// * `(image_names, species_ids)` - A tuple of string vectors.
#[pyfunction]
fn load_metadata(file_path: String) -> PyResult<(Vec<String>, Vec<String>)> {
    let df = if file_path.ends_with(".feather") {
        let file = std::fs::File::open(&file_path)?;
        IpcReader::new(file)
            .finish()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Polars IPC Error: {}", e)))?
    } else {
        let file = std::fs::File::open(&file_path)?;
        CsvReadOptions::default()
            .with_has_header(true)
            .with_parse_options(CsvParseOptions::default().with_separator(b';'))
            .into_reader_with_file_handle(file)
            .finish()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Polars CSV Error: {}", e)))?
    };

    // Extract columns with support for diverse naming conventions
    let image_name_s = df.column("image_name")
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Missing 'image_name' column: {}", e)))?;
    
    let species_ids_s = df.column("species_ids")
        .or_else(|_| df.column("species_id"))
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Metadata must contain 'species_ids' or 'species_id'"))?;

    // Safe string casting for mixed-type CSV columns
    let species_ids_str = species_ids_s.cast(&DataType::String)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Casting Error: {}", e)))?;

    let image_names = image_name_s.str()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .into_iter()
        .map(|opt_s| opt_s.unwrap_or("").to_string())
        .collect();

    let species_ids = species_ids_str.str()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .into_iter()
        .map(|opt_s| opt_s.unwrap_or("").to_string())
        .collect();

    Ok((image_names, species_ids))
}

/// Perform a 'Deep Audit' of a dataset using JWalk and Rayon.
///
/// 1. Crawls the physical disk in parallel to build an index of healthy files (>1KB).
/// 2. Intersects that index with the requested metadata paths.
///
/// # Arguments
/// * `img_dir` - The base directory to scan.
/// * `paths` - The target absolute paths to verify.
/// * `show_progress` - Whether to render an Indicatif progress bar.
#[pyfunction]
fn audit_dataset_jwalk(img_dir: String, paths: Vec<String>, show_progress: bool) -> PyResult<Vec<usize>> {
    if show_progress {
        println!("[Rust] Crawling directory tree: {}...", img_dir);
    }
    
    // Step 1: Parallel Directory Walk (Work-Stealing)
    let valid_files: HashSet<String> = WalkDir::new(&img_dir)
        .parallelism(jwalk::Parallelism::RayonNewPool(128))
        .into_iter()
        .filter_map(|entry| {
            if let Ok(ent) = entry {
                if ent.file_type().is_file() {
                    if let Ok(meta) = ent.metadata() {
                        // Integrity Filter: Ignore truncated or empty JPEGs
                        if meta.len() > 1024 {
                            return Some(ent.path().to_string_lossy().into_owned());
                        }
                    }
                }
            }
            None
        })
        .collect();

    if show_progress {
        println!("[Rust] Found {} healthy images. Intersecting with metadata...", valid_files.len());
    }

    // Step 2: Parallel Intersection
    let pb = if show_progress {
        let pb = ProgressBar::new(paths.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [Rust] Audit: [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"));
        Some(pb)
    } else {
        None
    };

    let valid_indices: Vec<usize> = paths
        .par_iter()
        .enumerate()
        .filter_map(|(idx, path_str)| {
            if let Some(ref pb) = pb { pb.inc(1); }
            if valid_files.contains(path_str) {
                Some(idx)
            } else {
                None
            }
        })
        .collect();

    if let Some(pb) = pb {
        pb.finish_with_message("Audit Complete");
    }

    Ok(valid_indices)
}

/// Legacy parallel scanner for specific path lists.
#[pyfunction]
fn audit_dataset(paths: Vec<String>, show_progress: bool) -> PyResult<Vec<usize>> {
    let pb = if show_progress {
        let pb = ProgressBar::new(paths.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [Rust] Verify: [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"));
        Some(pb)
    } else {
        None
    };

    let valid_indices: Vec<usize> = paths
        .par_iter()
        .enumerate()
        .filter_map(|(idx, path_str)| {
            if let Some(ref pb) = pb { pb.inc(1); }
            let path = Path::new(path_str);
            match fs::metadata(path) {
                Ok(meta) => {
                    if meta.is_file() && meta.len() > 1024 { Some(idx) } else { None }
                }
                Err(_) => None,
            }
        })
        .collect();

    if let Some(pb) = pb { pb.finish(); }
    Ok(valid_indices)
}

/// Bitset-based Taxonomic Impossible Combination Filter.
///
/// Moves complex hierarchical filtering from C++ to Rust for better 
/// memory safety and bit-level manipulation efficiency.
#[pyclass]
pub struct TaxonomicFilter {
    bitsets: Vec<BitVec<usize, Lsb0>>,
    species_to_genus: Vec<usize>,     // Hierarchy Mapping
    genus_to_family: Vec<usize>,    // Hierarchy Mapping
    bioclim_masks: Vec<BitVec<usize, Lsb0>>, // Bloom Filters for climate zones
    parent_map: Vec<usize>,          // Union-Find (DSU) structure
}

#[pymethods]
impl TaxonomicFilter {
    #[new]
    pub fn new(
        allowed_neighbors: Vec<Vec<usize>>,
        species_to_genus: Vec<usize>,
        genus_to_family: Vec<usize>,
        bioclim_data: Vec<Vec<usize>>, // List of species valid per climate cluster
    ) -> Self {
        let num_classes = allowed_neighbors.len();
        
        // 1. Arc Consistency Bitsets
        let mut bitsets = Vec::with_capacity(num_classes);
        for (i, neighbors) in allowed_neighbors.into_iter().enumerate() {
            let mut mask = bitvec![usize, Lsb0; 0; 8192];
            for n in neighbors {
                if n < 8192 { mask.set(n, true); }
            }
            if i < 8192 { mask.set(i, true); }
            bitsets.push(mask);
        }

        // 2. Bioclimatic Bloom Filters
        let mut bioclim_masks = Vec::with_capacity(bioclim_data.len());
        for cluster_species in bioclim_data {
            let mut mask = bitvec![usize, Lsb0; 0; 8192];
            for s in cluster_species {
                if s < 8192 { mask.set(s, true); }
            }
            bioclim_masks.push(mask);
        }

        TaxonomicFilter { 
            bitsets, 
            species_to_genus, 
            genus_to_family, 
            bioclim_masks,
            parent_map: Vec::new() // Initialized per quadrat
        }
    }

    /// Enforce Hierarchical Taxonomic Consistency (Species -> Genus -> Family).
    /// Prevents hallucinations where species signal contradicts genus confidence.
    pub fn enforce_hierarchy<'py>(
        &self,
        py: Python<'py>,
        probs: Bound<'py, PyArray1<f32>>,
        genus_probs: Bound<'py, PyArray1<f32>>,
        alpha: f32, // Weight of hierarchical boosting
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let mut p = unsafe { probs.as_array() }.to_owned();
        let g = unsafe { genus_probs.as_array() };
        
        for i in 0..p.len() {
            if i < self.species_to_genus.len() {
                let genus_idx = self.species_to_genus[i];
                if genus_idx < g.len() {
                    // Bayesian Boost: Multiply species prob by its genus confidence
                    // Branchless boost
                    let boost = g[genus_idx].powf(alpha);
                    p[i] *= boost;
                }
            }
        }
        Ok(p.to_pyarray_bound(py))
    }

    /// Bioclimatic Spatial Masking (Bloom Filter).
    /// Prunes species that have zero probability of occurring in a specific climate cluster.
    pub fn apply_spatial_mask<'py>(
        &self,
        py: Python<'py>,
        probs: Bound<'py, PyArray1<f32>>,
        cluster_id: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let mut p = unsafe { probs.as_array() }.to_owned();
        
        if cluster_id < self.bioclim_masks.len() {
            let mask = &self.bioclim_masks[cluster_id];
            for i in 0..p.len() {
                let is_valid = i < mask.len() && mask[i];
                p[i] *= is_valid as u32 as f32; // Branchless pruning
            }
        }
        Ok(p.to_pyarray_bound(py))
    }

    /// Union-Find (DSU) for Tile Clustering.
    /// Clusters tiles that likely contain the same physical plant to ensure 
    /// physical identity consistency across the quadrat.
    pub fn cluster_tiles(
        &mut self,
        similarities: Vec<Vec<f32>>,
        threshold: f32
    ) -> PyResult<Vec<usize>> {
        let n = similarities.len();
        self.parent_map = (0..n).collect();
        
        for i in 0..n {
            for j in (i+1)..n {
                if similarities[i][j] > threshold {
                    self.union(i, j);
                }
            }
        }
        
        // Return root assignments for each tile
        let mut roots = Vec::with_capacity(n);
        for i in 0..n {
            roots.push(self.find(i));
        }
        Ok(roots)
    }

    /// Frequency-Prior Laplace Smoothing.
    /// Bayesian prior adjustment to prevent "rare species hallucination".
    pub fn apply_laplace_priors<'py>(
        &self,
        py: Python<'py>,
        probs: Bound<'py, PyArray1<f32>>,
        frequencies: Vec<f32>,
        k: f32, // Smoothing constant
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let mut p = unsafe { probs.as_array() }.to_owned();
        let total_freq: f32 = frequencies.iter().sum();
        
        for i in 0..p.len() {
            if i < frequencies.len() {
                // Laplace smoothing: (count + k) / (total + k*num_classes)
                let prior = (frequencies[i] + k) / (total_freq + k * frequencies.len() as f32);
                p[i] *= prior.sqrt(); // Subtle damping
            }
        }
        Ok(p.to_pyarray_bound(py))
    }

    /// Arc Consistency (AC-3) Solver for Neuro-Symbolic Quadrat Reasoning.
    /// 
    /// Prunes species from tiles that are logically impossible given the 
    /// candidate set of neighboring tiles.
    pub fn solve_ac3<'py>(
        &self,
        py: Python<'py>,
        tile_probs: Vec<Bound<'py, PyArray1<f32>>>,
        threshold: f32,
    ) -> PyResult<Vec<Bound<'py, PyArray1<f32>>>> {
        let num_tiles = tile_probs.len();
        if num_tiles <= 1 { return Ok(tile_probs); }

        // 1. Initialize Domains (bitsets of candidates per tile)
        let mut domains: Vec<BitVec<usize, Lsb0>> = Vec::with_capacity(num_tiles);
        let mut original_data: Vec<Vec<f32>> = Vec::with_capacity(num_tiles);

        for p_arr in &tile_probs {
            let view = unsafe { p_arr.as_array() };
            let mut domain = bitvec![usize, Lsb0; 0; 8192];
            for (i, &p) in view.iter().enumerate() {
                if p > threshold && i < 8192 { domain.set(i, true); }
            }
            domains.push(domain);
            original_data.push(view.to_vec());
        }

        // 2. AC-3 Queue (all directed arcs between tiles)
        let mut queue = VecDeque::new();
        for i in 0..num_tiles {
            for j in 0..num_tiles {
                if i != j { queue.push_back((i, j)); }
            }
        }

        // 3. Constraint Propagation
        while let Some((i, j)) = queue.pop_front() {
            if self.revise(&mut domains, i, j) {
                // If Tile I's domain changed, re-check everyone who depends on it
                for k in 0..num_tiles {
                    if k != i { queue.push_back((k, i)); }
                }
            }
        }

        // 4. Return Pruned Probabilities
        let mut results = Vec::with_capacity(num_tiles);
        for (t, domain) in domains.iter().enumerate() {
            let mut p_vec = original_data[t].clone();
            for i in 0..p_vec.len() {
                if i < 8192 && !domain[i] {
                    p_vec[i] = 0.0;
                }
            }
            results.push(p_vec.to_pyarray_bound(py));
        }

        Ok(results)
    }

    /// Continuous Arc Consistency (AC-3) using EIVE + GBIF rules.
    ///
    /// $P'(A) = P(A) \times \min_{B} C_{AB}$
    /// where $C_{AB} = 1.0$ if GBIF co-occurrence > 0, else $C_{AB} = BC(A, B)$.
    pub fn solve_ac3_continuous<'py>(
        &self,
        py: Python<'py>,
        tile_probs: Vec<Bound<'py, PyArray1<f32>>>,
        threshold: f32,
        compatibility_matrix: Bound<'py, PyArray2<f32>>,
    ) -> PyResult<Vec<Bound<'py, PyArray1<f32>>>> {
        let num_tiles = tile_probs.len();
        if num_tiles == 0 { return Ok(tile_probs); }

        let comp_mat = unsafe { compatibility_matrix.as_array() };
        let num_classes = comp_mat.shape()[0];
        
        let mut original_data: Vec<Vec<f32>> = Vec::with_capacity(num_tiles);
        for p_arr in &tile_probs {
            original_data.push(unsafe { p_arr.as_array() }.to_vec());
        }

        // Find all species predicted across the entire quadrat (any tile > threshold)
        let mut predicted_species = Vec::new();
        for s in 0..num_classes {
            let mut is_predicted = false;
            for t in 0..num_tiles {
                if original_data[t][s] > threshold {
                    is_predicted = true;
                    break;
                }
            }
            if is_predicted {
                predicted_species.push(s);
            }
        }

        // For each tile, apply the continuous penalty
        let mut results = Vec::with_capacity(num_tiles);
        for t in 0..num_tiles {
            let mut p_vec = original_data[t].clone();
            
            for s_a in 0..num_classes {
                if p_vec[s_a] == 0.0 { continue; }
                
                // Find the worst compatibility conflict in the quadrat
                let mut min_c = 1.0f32;
                
                for &s_b in &predicted_species {
                    if s_a == s_b { continue; }
                    
                    // Check if GBIF co-occurrence > 0
                    let has_gbif = if s_a < self.bitsets.len() && s_b < self.bitsets[s_a].len() {
                        self.bitsets[s_a][s_b]
                    } else { false };
                    
                    let c_ab = if has_gbif {
                        1.0f32
                    } else {
                        if s_a < comp_mat.shape()[0] && s_b < comp_mat.shape()[1] {
                            comp_mat[[s_a, s_b]]
                        } else {
                            1.0f32
                        }
                    };
                    
                    if c_ab < min_c {
                        min_c = c_ab;
                    }
                }
                
                // Apply soft penalty
                p_vec[s_a] *= min_c;
            }
            results.push(p_vec.to_pyarray_bound(py));
        }

        Ok(results)
    }

    /// Suppress predictions that are ecologically or taxonomically impossible.
    /// Returns a new array with invalid class probabilities zeroed out.
    pub fn filter_predictions<'py>(
        &self, 
        py: Python<'py>, 
        predictions: Bound<'py, PyArray1<f32>>
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let array = unsafe { predictions.as_array() };
        let num_classes = array.len();
        
        let mut max_prob = -1.0;
        let mut anchor_idx = 0;
        
        // Find the most confident prediction to act as the taxonomic anchor
        for (i, &prob) in array.iter().enumerate() {
            if prob > max_prob {
                max_prob = prob;
                anchor_idx = i;
            }
        }
        
        let mut filtered = array.to_owned();
        
        if anchor_idx < self.bitsets.len() {
            let mask = &self.bitsets[anchor_idx];
            let len = mask.len();
            
            // ORACLE OPTIMIZATION: Branchless Validity Masking
            // Multiplies invalid species by 0.0, valid by 1.0. Allows CPU to auto-vectorize.
            for i in 0..num_classes {
                let is_valid = i < len && mask[i];
                let multiplier = is_valid as u32 as f32; // Branchless: true->1.0, false->0.0
                filtered[i] *= multiplier;
            }
        }
        
        Ok(filtered.to_pyarray_bound(py))
    }
}

/// Private helper implementation for TaxonomicFilter
impl TaxonomicFilter {
    fn find(&self, i: usize) -> usize {
        if self.parent_map[i] == i {
            i
        } else {
            self.find(self.parent_map[i])
        }
    }

    fn union(&mut self, i: usize, j: usize) {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i != root_j {
            self.parent_map[root_i] = root_j;
        }
    }

    /// REVISE(i, j): Prunes Tile I's domain based on Tile J's candidates.
    fn revise(&self, domains: &mut Vec<BitVec<usize, Lsb0>>, i: usize, j: usize) -> bool {
        let mut revised = false;
        
        // Optimization: Pre-calculate the union of all species allowed by Tile J
        let mut allowed_by_j = bitvec![usize, Lsb0; 0; 8192];
        for s_j in domains[j].iter_ones() {
            if s_j < self.bitsets.len() {
                allowed_by_j |= &self.bitsets[s_j];
            }
        }

        // Any species in Tile I NOT in allowed_by_j is logically impossible
        let domain_i = &mut domains[i];
        let old_count = domain_i.count_ones();
        *domain_i &= allowed_by_j;
        
        if domain_i.count_ones() < old_count {
            revised = true;
        }

        revised
    }
}

/// High-Performance Multi-Threaded Directory Indexer.
/// Returns a set of all filenames in a directory tree.
#[pyfunction]
fn index_directory_fast(dir_path: String) -> PyResult<HashSet<String>> {
    let files: HashSet<String> = WalkDir::new(&dir_path)
        .parallelism(jwalk::Parallelism::RayonNewPool(128))
        .into_iter()
        .filter_map(|entry| {
            if let Ok(ent) = entry {
                if ent.file_type().is_file() {
                    return Some(ent.file_name().to_string_lossy().into_owned());
                }
            }
            None
        })
        .collect();
    Ok(files)
}

/// Loopy Belief Propagation for Quadrat-Level Ecological Reasoning
/// 
/// Takes unary potentials (GFAM logits) for T tiles and a sparse pairwise 
/// potential matrix (ecological co-occurrence). Iteratively refines the 
/// marginal probability of each species in each tile based on its neighbors.
#[pyfunction]
fn loopy_belief_propagation<'py>(
    py: Python<'py>,
    unary_potentials: &Bound<'py, PyArray2<f32>>, // [num_tiles, num_species]
    pairwise_indices: Vec<(usize, usize)>,        // (species_a, species_b) co-occurrence
    pairwise_weights: Vec<f32>,                   // co-occurrence scores
    num_iterations: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    
    // Convert to ndarray view using pyo3 0.21+ API
    let unary = unsafe { unary_potentials.as_array() };
    let num_tiles = unary.shape()[0];
    let num_species = unary.shape()[1];

    // 1. Build adjacency list for sparse co-occurrence
    let mut species_adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_species];
    for ((s_a, s_b), &w) in pairwise_indices.into_iter().zip(pairwise_weights.iter()) {
        if s_a < num_species && s_b < num_species {
            species_adj[s_a].push((s_b, w));
            species_adj[s_b].push((s_a, w));
        }
    }

    // 2. Initialize messages: msg[source_tile][target_tile][species]
    let mut messages = vec![vec![vec![1.0f32; num_species]; num_tiles]; num_tiles];
    let mut next_messages = messages.clone();
    
    // 3. BP Iteration Loop
    for _iter in 0..num_iterations {
        // Parallelize across source tiles using Rayon
        next_messages.par_iter_mut().enumerate().for_each(|(j, node_msgs)| {
            for (i, msg_to_i) in node_msgs.iter_mut().enumerate() {
                if i == j { continue; }

                // product of incoming messages to j (excluding i)
                let mut prod_incoming = vec![1.0f32; num_species];
                for k in 0..num_tiles {
                    if k != i && k != j {
                        for s in 0..num_species {
                            prod_incoming[s] *= messages[k][j][s];
                        }
                    }
                }

                // Update message: m_{j->i}(s_i)
                for s_i in 0..num_species {
                    let mut max_val = 1e-12f32;
                    
                    // sparse traversal of co-occurrence edges
                    for &(s_j, weight) in &species_adj[s_i] {
                        let score = unary[[j, s_j]] * weight * prod_incoming[s_j];
                        if score > max_val { max_val = score; }
                    }
                    
                    // self-transition (identity)
                    let self_score = unary[[j, s_i]] * 1.0 * prod_incoming[s_i];
                    if self_score > max_val { max_val = self_score; }
                    
                    msg_to_i[s_i] = max_val;
                }

                // Normalization
                let sum: f32 = msg_to_i.iter().sum();
                if sum > 0.0 {
                    for val in msg_to_i.iter_mut() { *val /= sum; }
                }
            }
        });
        messages = next_messages.clone();
    }

    // 4. Compute final marginals
    let mut marginals = Array2::<f32>::zeros((num_tiles, num_species));
    for i in 0..num_tiles {
        let mut row = unary.row(i).to_owned();
        for j in 0..num_tiles {
            if i != j {
                for s in 0..num_species {
                    row[s] *= messages[j][i][s];
                }
            }
        }
        
        let sum = row.sum();
        if sum > 0.0 { row /= sum; }
        
        for s in 0..num_species {
            marginals[[i, s]] = row[s];
        }
    }

    Ok(marginals.to_pyarray_bound(py))
}

/// The `data_auditor` Python module.
#[pymodule]
fn data_auditor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(audit_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(audit_dataset_jwalk, m)?)?;
    m.add_function(wrap_pyfunction!(load_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(audit_compression_dna, m)?)?;
    m.add_function(wrap_pyfunction!(resize_image_lanczos, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_threshold_search, m)?)?;
    m.add_function(wrap_pyfunction!(sample_epoch_indices, m)?)?;
    m.add_function(wrap_pyfunction!(index_directory_fast, m)?)?;
    m.add_function(wrap_pyfunction!(apply_species_mask_avx512, m)?)?;
    m.add_function(wrap_pyfunction!(loopy_belief_propagation, m)?)?;
    m.add_class::<TaxonomicFilter>()?;
    m.add_class::<FeatureStore>()?;
    Ok(())
}
