use anyhow::Result;
use csv::ReaderBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{Read};
use std::path::{Path};
use tar::{Builder, Header};

// --- CONFIGURATION ---
const RESIZED_DIR: &str = "/workspace/plantclef/processed/train_700px/";
const CSV_PATH: &str = "/workspace/plantclef/processed/student_train_final.csv";
const OUTPUT_DIR: &str = "/workspace/plantclef/shards/";
const SHARD_SIZE: usize = 5000; // Number of images per 1GB shard
const MAX_THREADS: usize = 16;  // Optimized for concurrent shard writing

#[derive(Debug, Deserialize)]
struct Record {
    image_name: String,   // Changed from image_path to match CSV
    species_ids: String,  // Changed from species_id to match CSV
}

fn main() -> Result<()> {
    // 1. Setup
    fs::create_dir_all(OUTPUT_DIR)?;

    println!("[*] Reading metadata CSV...");
    let mut rdr = ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(true)
        .from_path(CSV_PATH)?;
    
    let records: Vec<Record> = rdr.deserialize()
        .filter_map(|r| {
            if let Err(ref e) = r {
                println!("[Warning] Deserialization error: {}", e);
            }
            r.ok()
        })
        .collect();
    
    let total_images = records.len();
    if total_images == 0 {
        println!("[Error] No images found in CSV! Please check the headers.");
        return Ok(());
    }
    
    let num_shards = (total_images + SHARD_SIZE - 1) / SHARD_SIZE;
    println!("[*] Found {} images. Packing into {} shards...", total_images, num_shards);

    // 2. Parallel Packing
    rayon::ThreadPoolBuilder::new().num_threads(MAX_THREADS).build_global().unwrap();
    
    let pb = ProgressBar::new(num_shards as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Shards (ETA: {eta})")?);

    (0..num_shards).into_par_iter().for_each(|shard_idx| {
        let _ = pack_shard(shard_idx, &records);
        pb.inc(1);
    });

    pb.finish_with_message("Done!");
    println!("[*] WebDataset shards created successfully at {}", OUTPUT_DIR);
    Ok(())
}

fn pack_shard(shard_idx: usize, records: &[Record]) -> Result<()> {
    let shard_path = Path::new(OUTPUT_DIR).join(format!("train_{:05}.tar", shard_idx));
    let file = File::create(shard_path)?;
    let mut tar = Builder::new(file);

    let start = shard_idx * SHARD_SIZE;
    let end = (start + SHARD_SIZE).min(records.len());

    for i in start..end {
        let record = &records[i];
        
        // Extract primary species ID (first one if comma-separated)
        let primary_sid = record.species_ids.split(',').next().unwrap_or("0").trim();
        let sid_int: u32 = primary_sid.parse().unwrap_or(0);

        // Path logic: student_train_final filenames are flat under the resized dir
        // or grouped by species depending on how they were saved.
        // We'll check both to be safe.
        let img_path_flat = Path::new(RESIZED_DIR).join(&record.image_name);
        let img_path_hier = Path::new(RESIZED_DIR).join(primary_sid).join(&record.image_name);
        
        let target_path = if img_path_flat.exists() {
            img_path_flat
        } else {
            img_path_hier
        };

        if let Ok(mut img_file) = File::open(&target_path) {
            let mut buffer = Vec::new();
            if img_file.read_to_end(&mut buffer).is_ok() {
                let stem = Path::new(&record.image_name)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("image");

                // A. Add JPEG
                let mut header = Header::new_gnu();
                header.set_size(buffer.len() as u64);
                header.set_path(format!("{}.jpg", stem))?;
                header.set_cksum();
                tar.append(&header, &buffer[..])?;

                // B. Add Label (.cls) to TAR in standard Text format
                let cls_data = sid_int.to_string().into_bytes(); 
                let mut cls_header = Header::new_gnu();
                cls_header.set_size(cls_data.len() as u64);
                cls_header.set_path(format!("{}.cls", stem))?;
                cls_header.set_cksum();
                tar.append(&cls_header, &cls_data[..])?;
            }
        }
    }

    tar.finish()?;
    Ok(())
}
