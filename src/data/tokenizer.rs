//! GPT-2 tokenizer
//! Converts from llmc/tokenizer.h

use crate::Result;
use std::fs::File;
use std::io::Read;
use std::io::Seek;
use std::path::Path;

pub struct Tokenizer {
    vocab_size: usize,
    token_table: Vec<String>,
    init_ok: bool,
}

impl Tokenizer {
    /// Load tokenizer from binary file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        use byteorder::{LittleEndian, ReadBytesExt};

        let mut file = File::open(path)?;

        // Read header
        let mut header = [0u32; 256];
        for i in 0..256 {
            header[i] = file.read_u32::<LittleEndian>()?;
        }

        // Validate magic number
        if header[0] != 20240328 {
            anyhow::bail!("Bad magic tokenizer file");
        }

        let version = header[1];
        let vocab_size = header[2] as usize;

        // Read token strings
        let mut token_table = Vec::with_capacity(vocab_size);

        if version == 1 {
            // Version 1: length-prefixed strings
            for _ in 0..vocab_size {
                let length = file.read_u8()? as usize;
                let mut token_bytes = vec![0u8; length];
                file.read_exact(&mut token_bytes)?;
                let token = String::from_utf8_lossy(&token_bytes).to_string();
                token_table.push(token);
            }
        } else if version == 2 {
            // Version 2: different format
            for _ in 0..vocab_size {
                let length = file.read_u32::<LittleEndian>()? as usize;
                let mut token_bytes = vec![0u8; length];
                file.read_exact(&mut token_bytes)?;
                let token = String::from_utf8_lossy(&token_bytes).to_string();
                token_table.push(token);
            }
        } else {
            anyhow::bail!("Unsupported tokenizer version: {}", version);
        }

        Ok(Self {
            vocab_size,
            token_table,
            init_ok: true,
        })
    }

    /// Decode a sequence of token IDs to a string
    pub fn decode(&self, tokens: &[i32]) -> String {
        let mut result = String::new();

        for &token_id in tokens {
            if token_id >= 0 && (token_id as usize) < self.vocab_size {
                result.push_str(&self.token_table[token_id as usize]);
            }
        }

        result
    }

    /// Encode a string to token IDs (simplified version)
    pub fn encode(&self, text: &str) -> Vec<i32> {
        // TODO: Implement proper BPE encoding
        // For now, return empty vector
        Vec::new()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
