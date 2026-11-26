// SPDX-License-Identifier: Apache-2.0
//! CPU uint8 bicubic antialias as used by the GLM torchvision processor.
//! Both passes round to bytes. Kernel precision is picked per axis from the
//! largest coefficient; Pillow's fixed precision can change the input bytes.
use std::borrow::Cow;

use llm_multimodal::vision::transforms::TransformError;

use super::pixels::{rgb_len, shape};

struct Kernel {
    rows: Vec<(usize, Vec<i32>)>,
    bits: u32,
}

fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else if x < 2.0 {
        -0.5 * (((x - 5.0) * x + 8.0) * x - 4.0)
    } else {
        0.0
    }
}

impl Kernel {
    fn new(src: usize, dst: usize) -> Self {
        let scale = src as f64 / dst as f64;
        let stretch = scale.max(1.0);
        let mut max_weight: f64 = 0.0;
        let rows = (0..dst)
            .map(|i| {
                let c = (i as f64 + 0.5) * scale;
                let start = ((c - 2.0 * stretch + 0.5) as usize).min(src);
                let end = ((c + 2.0 * stretch + 0.5) as usize).min(src);
                let mut taps =
                    (start..end).map(|j| cubic((j as f64 - c + 0.5) / stretch)).collect::<Vec<_>>();
                let sum: f64 = taps.iter().sum();
                for w in &mut taps {
                    *w /= sum;
                    max_weight = max_weight.max(*w);
                }
                (start, taps)
            })
            .collect::<Vec<_>>();
        let mut bits = 0;
        while bits < 22 && (max_weight * ((1u32 << (bits + 1)) as f64)).round() < 32768.0 {
            bits += 1;
        }
        let coeff_scale = (1u32 << bits) as f64;
        Self {
            bits,
            rows: rows
                .into_iter()
                .map(|(start, taps)| {
                    (
                        start,
                        taps.into_iter().map(|w| (w * coeff_scale).round() as i32).collect(),
                    )
                })
                .collect(),
        }
    }
    fn rgb(&self, src: &[u8], weights: &[i32], stride: usize, dest: &mut [u8]) {
        let mut sum = [1i64 << (self.bits - 1); 3];
        for (i, &weight) in weights.iter().enumerate() {
            let off = stride * i;
            for c in 0..3 {
                sum[c] += i64::from(src[off + c]) * i64::from(weight);
            }
        }
        for c in 0..3 {
            dest[c] = (sum[c] >> self.bits).clamp(0, 255) as u8;
        }
    }
}

pub(super) fn resize(
    src: &[u8],
    w: usize,
    h: usize,
    dw: usize,
    dh: usize,
) -> Result<Vec<u8>, TransformError> {
    if [w, h, dw, dh].contains(&0) || src.len() != rgb_len(w, h)? {
        return Err(shape("invalid GLM resize dimensions"));
    }
    let tmp = if w == dw {
        Cow::Borrowed(src)
    } else {
        let kernel = Kernel::new(w, dw);
        let mut tmp = vec![0; rgb_len(dw, h)?];
        for (src_row, row) in src.chunks_exact(w * 3).zip(tmp.chunks_exact_mut(dw * 3)) {
            for ((start, weights), pix) in kernel.rows.iter().zip(row.chunks_exact_mut(3)) {
                kernel.rgb(&src_row[start * 3..], weights, 3, pix);
            }
        }
        Cow::Owned(tmp)
    };
    if h == dh {
        return Ok(tmp.into_owned());
    }
    let kernel = Kernel::new(h, dh);
    let mut out = vec![0; rgb_len(dw, dh)?];
    for ((start, weights), row) in kernel.rows.iter().zip(out.chunks_exact_mut(dw * 3)) {
        for (x, pix) in row.chunks_exact_mut(3).enumerate() {
            kernel.rgb(&tmp[(start * dw + x) * 3..], weights, dw * 3, pix);
        }
    }
    Ok(out)
}
