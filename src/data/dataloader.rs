//! Data loader for reading tokenized datasets
//! Converts from llmc/dataloader.h

use crate::Result;
use memmap2::Mmap;
use ndarray::Array2;
use std::fs::File;
use std::path::Path;

pub struct DataLoader {
    tokens: Mmap,
    num_tokens: usize,
    batch_size: usize,
    seq_len: usize,
    current_position: usize,
}

impl DataLoader {
    /// Create a new data loader from a binary token file
    pub fn new<P: AsRef<Path>>(
        path: P,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Skip header (1024 bytes) and read tokens as u16
        let header_size = 1024;
        let num_tokens = (mmap.len() - header_size) / 2;

        Ok(Self {
            tokens: mmap,
            num_tokens,
            batch_size,
            seq_len,
            current_position: 0,
        })
    }

    /// Get the next batch of data
    pub fn next_batch(&mut self) -> Result<(Array2<i32>, Array2<i32>)> {
        use byteorder::{ByteOrder, LittleEndian};

        let header_size = 1024;
        let b = self.batch_size;
        let t = self.seq_len;

        let mut inputs = Array2::zeros((b, t));
        let mut targets = Array2::zeros((b, t));

        for i in 0..b {
            for j in 0..t {
                let pos = (self.current_position + i * t + j) % self.num_tokens;
                let byte_pos = header_size + pos * 2;

                let token = LittleEndian::read_u16(&self.tokens[byte_pos..byte_pos + 2]);
                inputs[[i, j]] = token as i32;

                // Target is next token
                let next_pos = (pos + 1) % self.num_tokens;
                let next_byte_pos = header_size + next_pos * 2;
                let next_token =
                    LittleEndian::read_u16(&self.tokens[next_byte_pos..next_byte_pos + 2]);
                targets[[i, j]] = next_token as i32;
            }
        }

        self.current_position = (self.current_position + b * t) % self.num_tokens;

        Ok((inputs, targets))
    }

    /// Reset the data loader to the beginning
    pub fn reset(&mut self) {
        self.current_position = 0;
    }

    /// Get the number of batches in the dataset
    pub fn num_batches(&self) -> usize {
        self.num_tokens / (self.batch_size * self.seq_len)
    }
}
