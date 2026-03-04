//! Utility functions for file I/O, memory management, and error handling

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Read a file and return its contents as bytes
pub fn read_file_bytes<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
    let mut file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open file: {:?}", path.as_ref()))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .with_context(|| format!("Failed to read file: {:?}", path.as_ref()))?;
    Ok(buffer)
}

/// Write bytes to a file
pub fn write_file_bytes<P: AsRef<Path>>(path: P, data: &[u8]) -> Result<()> {
    let mut file = File::create(path.as_ref())
        .with_context(|| format!("Failed to create file: {:?}", path.as_ref()))?;
    file.write_all(data)
        .with_context(|| format!("Failed to write to file: {:?}", path.as_ref()))?;
    Ok(())
}

/// Read a binary file containing f32 values
pub fn read_f32_binary<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
    use byteorder::{LittleEndian, ReadBytesExt};
    
    let mut file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open file: {:?}", path.as_ref()))?;
    
    let mut values = Vec::new();
    while let Ok(val) = file.read_f32::<LittleEndian>() {
        values.push(val);
    }
    
    Ok(values)
}

/// Write f32 values to a binary file
pub fn write_f32_binary<P: AsRef<Path>>(path: P, data: &[f32]) -> Result<()> {
    use byteorder::{LittleEndian, WriteBytesExt};
    
    let mut file = File::create(path.as_ref())
        .with_context(|| format!("Failed to create file: {:?}", path.as_ref()))?;
    
    for &val in data {
        file.write_f32::<LittleEndian>(val)
            .context("Failed to write f32 value")?;
    }
    
    Ok(())
}

/// Read i32 values from a binary file
pub fn read_i32_binary<P: AsRef<Path>>(path: P) -> Result<Vec<i32>> {
    use byteorder::{LittleEndian, ReadBytesExt};
    
    let mut file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open file: {:?}", path.as_ref()))?;
    
    let mut values = Vec::new();
    while let Ok(val) = file.read_i32::<LittleEndian>() {
        values.push(val);
    }
    
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_write_f32() {
        let data = vec![1.0f32, 2.5, 3.14, -1.5];
        let mut temp_file = NamedTempFile::new().unwrap();
        
        // Write using byteorder
        use byteorder::{LittleEndian, WriteBytesExt};
        for &val in &data {
            temp_file.write_f32::<LittleEndian>(val).unwrap();
        }
        temp_file.flush().unwrap();
        
        // Read back
        let read_data = read_f32_binary(temp_file.path()).unwrap();
        assert_eq!(data, read_data);
    }
}
