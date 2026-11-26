// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

// Direct mapping bounds file count and needs no shared index or directory scan.
// Reserve one extra slot's worth for an atomic replacement (including after a crash).
const MIN_SLOTS: usize = 256;
const MAX_SLOTS: usize = 16_384;
const FORMAT: &[u8] = b"vllm-rs-fastokens-0.2.1-segments-v1\0";
const DIGEST_BYTES: usize = 32;
pub(super) type Key = [u8; DIGEST_BYTES];
const POOL_ID_BYTES: usize = 16;

pub(super) struct SharedPool {
    dir: PathBuf,
    slots: usize,
    slot_bytes: usize,
    prefix: Vec<u8>,
}

impl SharedPool {
    pub(super) fn open(
        dir: PathBuf,
        budget: usize,
        namespace: String,
        tokenizer_json: &Value,
    ) -> io::Result<Self> {
        match fs::DirBuilder::new().mode(0o700).create(&dir) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
        if !dir.symlink_metadata()?.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "shared cache path must be a directory",
            ));
        }
        let dir = dir.canonicalize()?;
        let metadata = dir.metadata()?;
        if !metadata.is_dir() || metadata.permissions().mode() & 0o077 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "expected a private cache directory",
            ));
        }

        // Bind the pool to the complete effective tokenizer, including merged
        // added tokens. A changed model, format, or budget needs a fresh pool.
        let mut identity = Vec::from(FORMAT);
        identity.extend_from_slice(&(budget as u64).to_le_bytes());
        identity.extend_from_slice(&Sha256::digest(serde_json::to_vec(tokenizer_json)?));
        let identity_path = dir.join("identity");
        if !identity_path.try_exists()? {
            // Concurrent frontends may start together. Wait only during
            // pool initialization; request-path writes never wait.
            let lock = open_writer_lock(&dir)?;
            lock.lock()?;
            if !identity_path.try_exists()? {
                let mut record = vec![0; POOL_ID_BYTES];
                File::open("/dev/urandom")?.read_exact(&mut record)?;
                record.extend_from_slice(&identity);
                replace(&dir, &identity_path, &record)?;
            }
            drop(lock);
        }
        let mut actual = Vec::new();
        let mut file = File::open(identity_path)?;
        let mut pool_id = [0; POOL_ID_BYTES];
        file.read_exact(&mut pool_id)?;
        file.take(identity.len() as u64 + 1).read_to_end(&mut actual)?;
        if actual != identity {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "shared tokenizer identity or budget mismatch",
            ));
        }

        // A worker with a stale pool handle cannot populate a replacement
        // directory with records that its new workers would accept.
        let mut prefix = pool_id.to_vec();
        prefix.extend_from_slice(&(namespace.len() as u16).to_le_bytes());
        prefix.extend_from_slice(namespace.as_bytes());
        let slots = (budget / (1 << 20)).clamp(MIN_SLOTS, MAX_SLOTS);
        Ok(Self {
            dir,
            slots,
            slot_bytes: budget / (slots + 1),
            prefix,
        })
    }

    pub(super) fn key(&self, epoch: &str, mode: u8, text: &str) -> Option<Key> {
        // No request text, namespace or epoch is stored in clear on disk.
        // Skip hashing inputs too large to be practical as a shared entry.
        if text.len() > self.slot_bytes {
            return None;
        }
        let mut hash = Sha256::new();
        hash.update(&self.prefix);
        hash.update((epoch.len() as u16).to_le_bytes());
        hash.update(epoch.as_bytes());
        hash.update([mode]);
        hash.update(text.as_bytes());
        Some(hash.finalize().into())
    }

    pub(in crate::hf::encode_cache) fn slot_path(&self, key: &Key) -> PathBuf {
        let index = u32::from_le_bytes([key[0], key[1], key[2], key[3]]) as usize % self.slots;
        self.dir.join(format!("slot-{index:04x}"))
    }

    pub(super) fn get(&self, key: &Key) -> Option<Vec<u32>> {
        let mut raw = Vec::new();
        File::open(self.slot_path(key))
            .ok()?
            .take(self.slot_bytes as u64 + 1)
            .read_to_end(&mut raw)
            .ok()?;
        if raw.len() > self.slot_bytes {
            return None;
        }
        let (record, check) = raw.split_at_checked(raw.len().checked_sub(DIGEST_BYTES)?)?;
        let ids = record.strip_prefix(key.as_slice())?;
        let (values, remainder) = ids.as_chunks::<4>();
        let checksum: Key = Sha256::digest(record).into();
        if !remainder.is_empty() || checksum.as_slice() != check {
            return None;
        }
        Some(values.iter().copied().map(u32::from_le_bytes).collect())
    }

    pub(super) fn put(&self, key: &Key, ids: &[u32]) {
        let size = key.len() + size_of_val(ids) + DIGEST_BYTES;
        if size > self.slot_bytes {
            return;
        }
        let Ok(lock) = open_writer_lock(&self.dir) else {
            return;
        };
        if lock.try_lock().is_err() {
            return;
        }
        let mut record = Vec::with_capacity(size);
        record.extend_from_slice(key);
        for id in ids {
            record.extend_from_slice(&id.to_le_bytes());
        }
        let checksum = Sha256::digest(&record);
        record.extend_from_slice(&checksum);
        let _ = replace(&self.dir, &self.slot_path(key), &record);
    }
}

fn open_writer_lock(dir: &Path) -> io::Result<File> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(dir.join("writer.lock"))
}

fn replace(dir: &Path, path: &Path, bytes: &[u8]) -> io::Result<()> {
    // Only the lock owner touches this temporary name. Never modify a published
    // inode: concurrent readers see either complete version or a miss.
    let temp = dir.join("pending");
    let result = (|| {
        if temp.try_exists()? {
            fs::remove_file(&temp)?;
        }
        File::options()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temp)?
            .write_all(bytes)?;
        fs::rename(&temp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}
