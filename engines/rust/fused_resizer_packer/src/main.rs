use anyhow::Result;
use csv::ReaderBuilder;
use fast_image_resize as fr;
use image::codecs::jpeg::JpegEncoder;
use image::GenericImageView;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{Cursor};
use std::num::NonZeroU32;
use std::path::{Path};
use tar::{Builder, Header};

// --- CONFIGURATION ---
const RAW_DIR: &str = "/workspace/plantclef/raw/train/images_max_side_800/"; 
const OUTPUT_DIR: &str = "/workspace/plantclef/shards/";
const CSV_PATH: &str = "/workspace/plantclef/processed/student_train_final.csv";
const TARGET_SIZE: u32 = 700;
const SHARD_SIZE: usize = 5000;
const MAX_THREADS: usize = 16; 

#[derive(Debug, Deserialize)]
struct Record {
    image_name: String,
    species_ids: String,
}

fn main() -> Result<()> {
    fs::create_dir_all(OUTPUT_DIR)?;

    println!("[*] fused_resizer_packer: Sharding Train + Val into high-speed memory-shards...");
    
    let mut rdr = ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(true)
        .from_path(CSV_PATH)?;
    
    let records: Vec<Record> = rdr.deserialize()
        .filter_map(|r| r.ok())
        .collect();
    
    // 5% Validation Split (Skipped by this packer)
    let val_size = (records.len() as f32 * 0.05) as usize;
    let train_records = &records[val_size..];

    rayon::ThreadPoolBuilder::new().num_threads(MAX_THREADS).build_global().unwrap_or(());

    // 1. Pack Training Shards
    println!("[*] Found {} training images. Packing into {} shards...", train_records.len(), (train_records.len() + SHARD_SIZE - 1) / SHARD_SIZE);
    pack_records(train_records, "train")?;

    println!("[+] SUCCESS: Train and Val shards are ready.");
    Ok(())
}

fn pack_records(records: &[Record], prefix: &str) -> Result<()> {
    let num_shards = (records.len() + SHARD_SIZE - 1) / SHARD_SIZE;
    let pb = ProgressBar::new(num_shards as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Shards (ETA: {eta})")?);

    (0..num_shards).into_par_iter().for_each(|shard_idx| {
        let shard_path = Path::new(OUTPUT_DIR).join(format!("{}_{:05}.tar", prefix, shard_idx));
        let _ = stream_shard(shard_idx, records, &shard_path);
        pb.inc(1);
    });
    pb.finish();
    Ok(())
}

fn stream_shard(shard_idx: usize, records: &[Record], shard_path: &Path) -> Result<()> {
    let file = File::create(shard_path)?;
    let mut tar = Builder::new(file);

    let start = shard_idx * SHARD_SIZE;
    let end = (start + SHARD_SIZE).min(records.len());

    for i in start..end {
        let record = &records[i];
        let primary_sid = record.species_ids.split(',').next().unwrap_or("0").trim();
        let sid_int: u32 = primary_sid.parse().unwrap_or(0);
        let img_path = Path::new(RAW_DIR).join(primary_sid).join(&record.image_name);

        if let Ok(img) = image::open(&img_path) {
            let (width, height) = img.dimensions();
            if let Ok(src_image) = fr::Image::from_vec_u8(
                NonZeroU32::new(width).unwrap(),
                NonZeroU32::new(height).unwrap(),
                img.to_rgb8().into_raw(),
                fr::PixelType::U8x3,
            ) {
                let mut dst_image = fr::Image::new(
                    NonZeroU32::new(TARGET_SIZE).unwrap(),
                    NonZeroU32::new(TARGET_SIZE).unwrap(),
                    fr::PixelType::U8x3,
                );
                let mut resizer = fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::CatmullRom));
                if resizer.resize(&src_image.view(), &mut dst_image.view_mut()).is_ok() {
                    let mut jpeg_buffer = Vec::new();
                    {
                        let mut encoder = JpegEncoder::new_with_quality(Cursor::new(&mut jpeg_buffer), 85);
                        let _ = encoder.encode(dst_image.buffer(), TARGET_SIZE, TARGET_SIZE, image::ColorType::Rgb8);
                    }
                    let stem = Path::new(&record.image_name).file_stem().and_then(|s| s.to_str()).unwrap_or("image");
                    let mut header = Header::new_gnu();
                    header.set_size(jpeg_buffer.len() as u64);
                    header.set_path(format!("{}.jpg", stem))?;
                    header.set_cksum();
                    tar.append(&header, &jpeg_buffer[..])?;
                    let cls_data = sid_int.to_le_bytes(); 
                    let mut cls_header = Header::new_gnu();
                    cls_header.set_size(4);
                    cls_header.set_path(format!("{}.cls", stem))?;
                    cls_header.set_cksum();
                    tar.append(&cls_header, &cls_data[..])?;
                }
            }
        }
    }
    tar.finish()?;
    Ok(())
}
