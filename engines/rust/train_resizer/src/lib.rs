use fast_image_resize as fr;
use image::GenericImageView;
use std::ffi::CStr;
use std::num::NonZeroU32;
use std::os::raw::{c_char, c_int};
use std::ptr;

/**
 * @brief C-FFI for the train-set image resizer.
 *
 * Allows C++ callers to invoke the Rust SIMD resize pipeline.
 */

#[repr(C)]
pub struct ResizedImage {
    pub data: *mut u8,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

#[no_mangle]
pub extern "C" fn rust_resize_image(
    path: *const c_char,
    target_size: u32,
) -> ResizedImage {
    let c_str = unsafe {
        if path.is_null() { return ResizedImage { data: ptr::null_mut(), width: 0, height: 0, channels: 0 }; }
        CStr::from_ptr(path)
    };

    let str_path = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ResizedImage { data: ptr::null_mut(), width: 0, height: 0, channels: 0 },
    };

    // 1. Decode
    let img = match image::open(str_path) {
        Ok(i) => i,
        Err(_) => return ResizedImage { data: ptr::null_mut(), width: 0, height: 0, channels: 0 },
    };

    let (width, height) = img.dimensions();

    // 2. Scale
    let (n_width, n_height) = if width > height {
        (target_size, (target_size as f32 * (height as f32 / width as f32)) as u32)
    } else {
        ((target_size as f32 * (width as f32 / height as f32)) as u32, target_size)
    };

    // 3. Resize
    let src_image = match fr::Image::from_vec_u8(
        NonZeroU32::new(width).unwrap(),
        NonZeroU32::new(height).unwrap(),
        img.to_rgb8().into_raw(),
        fr::PixelType::U8x3,
    ) {
        Ok(si) => si,
        Err(_) => return ResizedImage { data: ptr::null_mut(), width: 0, height: 0, channels: 0 },
    };

    let mut dst_image = fr::Image::new(
        NonZeroU32::new(n_width).unwrap(),
        NonZeroU32::new(n_height).unwrap(),
        src_image.pixel_type(),
    );

    let mut resizer = fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::CatmullRom));
    if resizer.resize(&src_image.view(), &mut dst_image.view_mut()).is_err() {
        return ResizedImage { data: ptr::null_mut(), width: 0, height: 0, channels: 0 };
    }

    // 4. Return raw buffer (Note: Box::into_raw to leak to C++ memory space)
    let buffer = dst_image.into_vec();
    let width = n_width;
    let height = n_height;
    let channels = 3;
    
    let mut boxed_slice = buffer.into_boxed_slice();
    let data_ptr = boxed_slice.as_mut_ptr();
    std::mem::forget(boxed_slice); // Hand over ownership to C++

    ResizedImage {
        data: data_ptr,
        width,
        height,
        channels,
    }
}

#[no_mangle]
pub extern "C" fn rust_free_image(data: *mut u8, length: usize) {
    if !data.is_null() {
        unsafe {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(data, length));
        }
    }
}
