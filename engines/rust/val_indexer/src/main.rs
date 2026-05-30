use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

fn parse_octal(bytes: &[u8]) -> u64 {
    let mut val = 0;
    for &b in bytes {
        if b >= b'0' && b <= b'7' {
            val = val * 8 + (b - b'0') as u64;
        } else if b == 0 || b == b' ' {
            if val > 0 { break; }
        }
    }
    val
}

struct Component {
    offset: u64,
    size: u64,
    filename: String,
}

struct Sample {
    jpg: Component,
    cls: Component,
}

fn generate_index(tar_path: &Path, idx_path: &Path) -> Result<()> {
    let mut file = File::open(tar_path)?;
    let mut header = [0u8; 512];
    let mut samples = Vec::with_capacity(5000);
    
    let mut m1_comp: Option<Component> = None;
    
    loop {
        let offset = file.stream_position()?;
        let bytes_read = file.read(&mut header)?;
        if bytes_read < 512 || header[0] == 0 {
            break; // EOF
        }
        
        let name_end = header[0..100].iter().position(|&b| b == 0).unwrap_or(100);
        let full_name = std::str::from_utf8(&header[0..name_end])?.to_string();
        let size = parse_octal(&header[124..136]);
        let padded_size = (size + 511) / 512 * 512;
        let ext = if full_name.ends_with(".jpg") { "jpg" } else { "cls" };
        
        let comp = Component {
            offset: offset + 512,
            size,
            filename: full_name,
        };

        if ext == "jpg" {
            m1_comp = Some(comp);
        } else if let Some(j) = m1_comp.take() {
            samples.push(Sample { jpg: j, cls: comp });
        }
        file.seek(SeekFrom::Current(padded_size as i64))?;
    }

    let mut out = String::with_capacity(1 * 1024 * 1024);
    out.push_str(&format!("v1.2 {}\n", samples.len()));
    for s in samples {
        out.push_str(&format!(
            "jpg {} {} {} cls {} {} {}\n", 
            s.jpg.offset, s.jpg.size, s.jpg.filename,
            s.cls.offset, s.cls.size, s.cls.filename
        ));
    }
    std::fs::write(idx_path, out)?;
    Ok(())
}

fn main() -> Result<()> {
    let shard_dir = Path::new("/workspace/plantclef/shards");
    let index_dir = Path::new("/workspace/plantclef/shards/.index");
    
    std::fs::create_dir_all(index_dir)?;
    
    let mut tar_files: Vec<PathBuf> = std::fs::read_dir(shard_dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            name.starts_with("val_") && p.extension().and_then(|s| s.to_str()) == Some("tar")
        })
        .collect();
    tar_files.sort();
    
    if tar_files.is_empty() {
        println!("[Error] No validation shards (val_*.tar) found in {}", shard_dir.display());
        return Ok(());
    }

    println!("[*] Found {} validation shards. Generating indices...", tar_files.len());
    let pb = ProgressBar::new(tar_files.len() as u64);
    pb.set_style(ProgressStyle::with_template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Shards").unwrap());
        
    tar_files.par_iter().for_each(|tar_path| {
        let stem = tar_path.file_stem().unwrap().to_str().unwrap();
        let idx_path = index_dir.join(format!("{}.idx", stem));
        let _ = generate_index(tar_path, &idx_path);
        pb.inc(1);
    });
    
    pb.finish_with_message("Done!");
    println!("[+] SUCCESS: Validation indices generated.");
    Ok(())
}
