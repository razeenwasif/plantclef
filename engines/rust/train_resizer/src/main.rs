use anyhow::Result;
use fast_image_resize as fr;
use image::codecs::jpeg::JpegEncoder;
use image::GenericImageView;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

// --- CONFIGURATION ---
const SRC_DIR: &str = "/workspace/plantclef/raw/train/";
const DST_DIR: &str = "/workspace/plantclef/processed/train_700px/";
const TARGET_SIZE: u32 = 700;
const JPEG_QUALITY: u8 = 85;
const MAX_THREADS: usize = 16; // Throttled to prevent Network Volume congestion

fn main() -> Result<()> {
    let src_path = Path::new(SRC_DIR);
    let dst_path = Path::new(DST_DIR);

    if !src_path.exists() {
        println!("[Error] Source directory not found: {}", SRC_DIR);
        return Ok(());
    }

    println!("[*] Scanning directory for 1.4M images...");
    let entries: Vec<PathBuf> = WalkDir::new(src_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let ext = e.path().extension().and_then(|s| s.to_str()).unwrap_or("");
            ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg")
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    println!("[*] Found {} images. Throttling Rayon to {} workers for I/O stability...", entries.len(), MAX_THREADS);

    // Explicitly configure thread pool for Network Volume efficiency
    rayon::ThreadPoolBuilder::new().num_threads(MAX_THREADS).build_global().unwrap();

    let pb = ProgressBar::new(entries.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec} | ETA: {eta})")?
        .progress_chars("#>-"));

    entries.par_iter().for_each(|img_path| {
        if let Err(_e) = process_single_image(img_path, src_path, dst_path) {
            // Errors skipped to maintain swarm momentum
        }
        pb.inc(1);
    });

    pb.finish_with_message("Done!");
    Ok(())
}

fn process_single_image(img_path: &Path, src_root: &Path, dst_root: &Path) -> Result<()> {
    let rel_path = img_path.strip_prefix(src_root)?;
    let save_path = dst_root.join(rel_path);

    if save_path.exists() {
        return Ok(());
    }

    if let Some(parent) = save_path.parent() {
        fs::create_dir_all(parent).unwrap_or(());
    }

    // 1. Buffered Decode (Minimizes network packets)
    let file = File::open(img_path)?;
    let reader = BufReader::with_capacity(128 * 1024, file);
    let img = image::load(reader, image::ImageFormat::Jpeg)?;
    
    let (width, height) = img.dimensions();

    // 2. Scale calculations
    let (n_width, n_height) = if width > height {
        (TARGET_SIZE, (TARGET_SIZE as f32 * (height as f32 / width as f32)) as u32)
    } else {
        ((TARGET_SIZE as f32 * (width as f32 / height as f32)) as u32, TARGET_SIZE)
    };

    // 3. RGB8 Resize (25% less data than RGBA)
    let src_image = fr::Image::from_vec_u8(
        NonZeroU32::new(width).unwrap(),
        NonZeroU32::new(height).unwrap(),
        img.to_rgb8().into_raw(),
        fr::PixelType::U8x3,
    )?;

    let mut dst_image = fr::Image::new(
        NonZeroU32::new(n_width).unwrap(),
        NonZeroU32::new(n_height).unwrap(),
        src_image.pixel_type(),
    );

    let mut resizer = fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::CatmullRom));
    resizer.resize(&src_image.view(), &mut dst_image.view_mut())?;

    // 4. Buffered Encode
    let result_file = File::create(save_path)?;
    let writer = BufWriter::with_capacity(128 * 1024, result_file);
    let mut encoder = JpegEncoder::new_with_quality(writer, JPEG_QUALITY);
    encoder.encode(
        dst_image.buffer(),
        n_width,
        n_height,
        image::ColorType::Rgb8,
    )?;

    Ok(())
}
