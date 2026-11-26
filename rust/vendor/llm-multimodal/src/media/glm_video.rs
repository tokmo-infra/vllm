//! GLM5Next loader: select exact source indices and derive prompt timestamps
//! from the SAME indices. Decode only unique picks; the pixel packer pads the tail.
use super::*;

fn invalid(message: &str) -> MediaConnectorError {
    MediaConnectorError::InvalidVideo(message.to_owned())
}

fn sample(total: usize, fps: f64, interval: f64, cap: usize, temporal: usize) -> Vec<usize> {
    if total == 0
        || !fps.is_finite()
        || fps <= 0.
        || !interval.is_finite()
        || interval <= 0.
        || temporal == 0
    {
        return vec![];
    }
    let duration = total as f64 / fps;
    let n = ((duration * interval) as usize).min(cap);
    if n == 0 {
        return vec![];
    }
    let mut picks = Vec::with_capacity(n);
    if total < n {
        picks.extend((0..n).map(|i| ((i as u128 * total as u128) / n as u128) as usize));
    } else {
        let mut next = 0.;
        let step = 1. / (temporal as f64 * interval);
        let frame_time = 1. / fps;
        let mut start = 0;
        while start < total && picks.len() <= n {
            // Equivalent to walking every frame, without work proportional to an
            // untrusted container header's frame count. Compare the exact same
            // floating point expression as the training implementation.
            let (mut lo, mut hi) = (start, total);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if mid as f64 * frame_time >= next {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            if lo == total {
                break;
            }
            picks.push(lo);
            start = lo + 1;
            next += step;
            if next >= duration.floor() {
                break;
            }
        }
        if picks.len() != n {
            let last = if picks.len() < n {
                picks.last().copied().unwrap_or(total - 1)
            } else {
                total - 1
            };
            let stride = last as f64 / n.saturating_sub(1).max(1) as f64;
            picks.clear();
            picks.extend((0..n).map(|i| {
                if i > 0 && i + 1 == n {
                    last
                } else {
                    (i as f64 * stride).floor() as usize
                }
            }));
        }
    }
    picks.dedup();
    picks
}

// Balanced AST rather than a left-nested sum for clips with thousands of picks.
fn selector(indices: &[usize]) -> String {
    if indices.len() == 1 {
        return format!("eq(n,{})", indices[0]);
    }
    let (l, r) = indices.split_at(indices.len() / 2);
    format!("({}+{})", selector(l), selector(r))
}

pub(super) async fn decode(
    bytes: Bytes,
    temporal: usize,
    interval: f64,
    cap: usize,
) -> Result<(DecodedRgbVideo, Vec<i64>), MediaConnectorError> {
    let tmp = task::spawn_blocking(move || write_temp_video_file(&bytes)).await??;
    let info = probe_video_info(tmp.path()).await?;
    let fps = info.fps.ok_or_else(|| invalid("missing source video frame rate"))?;
    let total = info
        .frames
        .or_else(|| info.duration_seconds.map(|d| (d * fps).round() as u32))
        .ok_or_else(|| invalid("missing source video frame count"))?;
    let picks = sample(total as usize, fps, interval, cap, temporal);
    if picks.is_empty() {
        return Err(invalid(
            "GLM-5-Next frame sampler selected no frames: clip shorter than one sampling interval",
        ));
    }
    let width = info.width.ok_or_else(|| invalid("missing video width"))?;
    let height = info.height.ok_or_else(|| invalid("missing video height"))?;
    let size = checked_decoded_rgb_bytes(picks.len(), rawvideo_frame_size(width, height)?)?;
    let limit = size
        .checked_add(picks.len() * 64)
        .ok_or_else(|| invalid("sampled RGB size overflow"))?;
    let mut command = Command::new("ffmpeg");
    command
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
        .arg(tmp.path())
        .args([
            "-map",
            "0:v:0",
            "-vf",
            &format!("select='{}'", selector(&picks)),
            "-vsync",
            "0",
            "-frames:v",
            &picks.len().to_string(),
            "-fs",
            &limit.to_string(),
            "-threads",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "ppm",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]);
    let out = run_video_command_output(command, "ffmpeg").await?;
    if !out.status.success() {
        return Err(invalid(&format!(
            "ffmpeg frame selection failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let rgb = parse_ppm_rgb_video(out.stdout.into())?;
    if rgb.frames.len() != picks.len() {
        return Err(invalid(
            "decoded frame count disagrees with source index selection",
        ));
    }
    let seconds = picks.iter().step_by(temporal).map(|&idx| (idx as f64 / fps) as i64).collect();
    Ok((rgb, seconds))
}
#[cfg(test)]
mod tests {
    use super::sample;
    #[test]
    fn python_source_index_goldens() {
        for (n, fps, cap, want) in [
            (3, 1., 2048, vec![0, 1, 2]),
            (31, 29.97, 2048, vec![0, 30]),
            (121, 29.97, 2048, vec![0, 17, 34, 51, 68, 85, 102, 120]),
            (48, 8., 5, vec![0, 11, 23, 35, 47]),
            (11, 3.2, 2048, vec![0, 2, 4, 6, 8, 10]),
            (4, 15., 2048, vec![]),
            (3, 6., 2048, vec![0]),
        ] {
            assert_eq!(sample(n, fps, 2., cap, 2), want);
        }
    }
}
