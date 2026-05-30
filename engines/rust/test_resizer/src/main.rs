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
const SRC_DIR: &str = "/workspace/plantclef/raw/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images/";
const DST_DIR: &str = "/workspace/plantclef/processed/test_700px/";
const TARGET_SIZE: u32 = 700;
const JPEG_QUALITY: u8 = 90;
const MAX_THREADS: usize = 16; 

fn main() -> Result<()> {
    let src_path = Path::new(SRC_DIR);
    let dst_path = Path::new(DST_DIR);

    if !src_path.exists() {
        println!("[Error] Test source directory not found: {}", SRC_DIR);
        return Ok(());
    }

    fs::create_dir_all(dst_path)?;

    println!("[*] Scanning test directory for 2,105 high-res images...");
    let entries: Vec<PathBuf> = WalkDir::new(src_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let ext = e.path().extension().and_then(|s| s.to_str()).unwrap_or("");
            ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg")
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    println!("[*] Found {} images. Launching test-set resizer...", entries.len());

    rayon::ThreadPoolBuilder::new().num_threads(MAX_THREADS).build_global().unwrap_or(());
    
    let pb = ProgressBar::new(entries.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} (ETA: {eta})")?);

    entries.par_iter().for_each(|img_path| {
        let _ = process_test_image(img_path, src_path, dst_path);
        pb.inc(1);
    });

    pb.finish_with_message("Done!");
    println!("[*] Test set resized successfully to {}", DST_DIR);
    Ok(())
}

fn process_test_image(img_path: &Path, src_root: &Path, dst_root: &Path) -> Result<()> {
    let rel_path = img_path.strip_prefix(src_root)?;
    let save_path = dst_root.join(rel_path);

    if save_path.exists() {
        return Ok(());
    }

    if let Some(parent) = save_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // 1. Buffered Decode
    let file = File::open(img_path)?;
    let reader = BufReader::with_capacity(256 * 1024, file);
    let img = image::load(reader, image::ImageFormat::Jpeg)?;
    
    let (width, height) = img.dimensions();
    let (n_width, n_height) = if width > height {
        (TARGET_SIZE, (TARGET_SIZE as f32 * (height as f32 / width as f32)) as u32)
    } else {
        ((TARGET_SIZE as f32 * (width as f32 / height as f32)) as u32, TARGET_SIZE)
    };

    // 2. High-Precision Resize (Lanczos3 for Inference quality)
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

    let mut resizer = fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::Lanczos3));
    resizer.resize(&src_image.view(), &mut dst_image.view_mut())?;

    // 3. Encode (High quality for expert identification)
    let result_file = File::create(save_path)?;
    let writer = BufWriter::with_capacity(256 * 1024, result_file);
    let mut encoder = JpegEncoder::new_with_quality(writer, JPEG_QUALITY);
    encoder.encode(
        dst_image.buffer(),
        n_width,
        n_height,
        image::ColorType::Rgb8,
    )?;

    Ok(())
}
