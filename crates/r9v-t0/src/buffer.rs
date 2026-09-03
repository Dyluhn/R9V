// SPDX-License-Identifier: Apache-2.0
//! Buffer and view abstractions for scalar T0 tensor operands (Spec 1 §2.3, Spec 4 §2).

use r9v_ir::DType;

use crate::dtype::{
    bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, fp8_e4m3_decode, fp8_e4m3_encode,
    fp8_e5m2_decode, fp8_e5m2_encode, read_f32_at, read_f64_at, write_f32_at,
};

/// Borrowed tensor data slice variant without raw pointer conversions.
#[derive(Debug, Clone)]
pub enum TensorData<'a> {
    /// 32-bit floating point slice.
    F32(&'a [f32]),
    /// 16-bit half precision float slice.
    F16(&'a [u16]),
    /// 16-bit bfloat16 slice.
    Bf16(&'a [u16]),
    /// 8-bit signed integer slice.
    I8(&'a [i8]),
    /// 32-bit unsigned integer slice.
    U32(&'a [u32]),
    /// Raw byte slice with associated data type.
    Bytes(DType, &'a [u8]),
}

/// Mutable borrowed tensor data slice variant without raw pointer conversions.
#[derive(Debug)]
pub enum TensorDataMut<'a> {
    /// Mutable 32-bit floating point slice.
    F32(&'a mut [f32]),
    /// Mutable 16-bit half precision float slice.
    F16(&'a mut [u16]),
    /// Mutable 16-bit bfloat16 slice.
    Bf16(&'a mut [u16]),
    /// Mutable 8-bit signed integer slice.
    I8(&'a mut [i8]),
    /// Mutable 32-bit unsigned integer slice.
    U32(&'a mut [u32]),
    /// Mutable raw byte slice with associated data type.
    Bytes(DType, &'a mut [u8]),
}

/// Immutable view over a multidimensional tensor buffer.
#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    shape: Vec<usize>,
    data: TensorData<'a>,
}

impl<'a> TensorView<'a> {
    /// Creates a tensor view from raw bytes with shape and dtype.
    pub fn from_bytes(shape: &[usize], dtype: DType, data: &'a [u8]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: TensorData::Bytes(dtype, data),
        }
    }

    /// Creates a tensor view from an `f32` slice.
    pub fn from_f32_slice(shape: &[usize], data: &'a [f32]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: TensorData::F32(data),
        }
    }

    /// Creates a tensor view from an `f16` (u16 bits) slice.
    pub fn from_f16_slice(shape: &[usize], data: &'a [u16]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: TensorData::F16(data),
        }
    }

    /// Creates a tensor view from an `bf16` (u16 bits) slice.
    pub fn from_bf16_slice(shape: &[usize], data: &'a [u16]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: TensorData::Bf16(data),
        }
    }

    /// Creates a tensor view from an `i8` slice.
    pub fn from_i8_slice(shape: &[usize], data: &'a [i8]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: TensorData::I8(data),
        }
    }

    /// Creates a tensor view from a `u32` slice.
    pub fn from_u32_slice(shape: &[usize], data: &'a [u32]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: TensorData::U32(data),
        }
    }

    /// Returns tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns tensor rank.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns tensor data type.
    pub fn dtype(&self) -> DType {
        match self.data {
            TensorData::F32(_) => DType::F32,
            TensorData::F16(_) => DType::F16,
            TensorData::Bf16(_) => DType::Bf16,
            TensorData::I8(_) => DType::I8,
            TensorData::U32(_) => DType::U32,
            TensorData::Bytes(dtype, _) => dtype,
        }
    }

    /// Returns total number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Reads one element converted to `f32`.
    pub fn read_f32(&self, index: usize) -> f32 {
        match self.data {
            TensorData::F32(slice) => slice[index],
            TensorData::F16(slice) => f16_to_f32(slice[index]),
            TensorData::Bf16(slice) => bf16_to_f32(slice[index]),
            TensorData::I8(slice) => slice[index] as f32,
            TensorData::U32(slice) => slice[index] as f32,
            TensorData::Bytes(dtype, slice) => read_f32_at(dtype, slice, index),
        }
    }

    /// Reads one element converted to `f64`.
    pub fn read_f64(&self, index: usize) -> f64 {
        match self.data {
            TensorData::F32(slice) => slice[index] as f64,
            TensorData::F16(slice) => f16_to_f32(slice[index]) as f64,
            TensorData::Bf16(slice) => bf16_to_f32(slice[index]) as f64,
            TensorData::I8(slice) => slice[index] as f64,
            TensorData::U32(slice) => slice[index] as f64,
            TensorData::Bytes(dtype, slice) => read_f64_at(dtype, slice, index),
        }
    }

    /// Reads one element as `u32`.
    pub fn read_u32(&self, index: usize) -> u32 {
        match self.data {
            TensorData::U32(slice) => slice[index],
            _ => self.read_f32(index) as u32,
        }
    }

    /// Reads one element as `i8`.
    pub fn read_i8(&self, index: usize) -> i8 {
        match self.data {
            TensorData::I8(slice) => slice[index],
            _ => self.read_f32(index) as i8,
        }
    }
}

/// Mutable view over a multidimensional tensor buffer.
#[derive(Debug)]
pub struct TensorViewMut<'a> {
    shape: Vec<usize>,
    data: TensorDataMut<'a>,
}

impl<'a> TensorViewMut<'a> {
    /// Creates a mutable tensor view from raw bytes with shape and dtype.
    pub fn from_bytes(shape: &[usize], dtype: DType, data: &'a mut [u8]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: TensorDataMut::Bytes(dtype, data),
        }
    }

    /// Creates a mutable tensor view from an `f32` slice.
    pub fn from_f32_slice(shape: &[usize], data: &'a mut [f32]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: TensorDataMut::F32(data),
        }
    }

    /// Creates a mutable tensor view from an `f16` (u16 bits) slice.
    pub fn from_f16_slice(shape: &[usize], data: &'a mut [u16]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: TensorDataMut::F16(data),
        }
    }

    /// Creates a mutable tensor view from an `bf16` (u16 bits) slice.
    pub fn from_bf16_slice(shape: &[usize], data: &'a mut [u16]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: TensorDataMut::Bf16(data),
        }
    }

    /// Creates a mutable tensor view from an `i8` slice.
    pub fn from_i8_slice(shape: &[usize], data: &'a mut [i8]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: TensorDataMut::I8(data),
        }
    }

    /// Creates a mutable tensor view from a `u32` slice.
    pub fn from_u32_slice(shape: &[usize], data: &'a mut [u32]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: TensorDataMut::U32(data),
        }
    }

    /// Returns tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns tensor rank.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns tensor data type.
    pub fn dtype(&self) -> DType {
        match self.data {
            TensorDataMut::F32(_) => DType::F32,
            TensorDataMut::F16(_) => DType::F16,
            TensorDataMut::Bf16(_) => DType::Bf16,
            TensorDataMut::I8(_) => DType::I8,
            TensorDataMut::U32(_) => DType::U32,
            TensorDataMut::Bytes(dtype, _) => dtype,
        }
    }

    /// Returns total number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Reads one element converted to `f32`.
    pub fn read_f32(&self, index: usize) -> f32 {
        match self.data {
            TensorDataMut::F32(ref slice) => slice[index],
            TensorDataMut::F16(ref slice) => f16_to_f32(slice[index]),
            TensorDataMut::Bf16(ref slice) => bf16_to_f32(slice[index]),
            TensorDataMut::I8(ref slice) => slice[index] as f32,
            TensorDataMut::U32(ref slice) => slice[index] as f32,
            TensorDataMut::Bytes(dtype, ref slice) => read_f32_at(dtype, slice, index),
        }
    }

    /// Reads one element converted to `f64`.
    pub fn read_f64(&self, index: usize) -> f64 {
        match self.data {
            TensorDataMut::F32(ref slice) => slice[index] as f64,
            TensorDataMut::F16(ref slice) => f16_to_f32(slice[index]) as f64,
            TensorDataMut::Bf16(ref slice) => bf16_to_f32(slice[index]) as f64,
            TensorDataMut::I8(ref slice) => slice[index] as f64,
            TensorDataMut::U32(ref slice) => slice[index] as f64,
            TensorDataMut::Bytes(dtype, ref slice) => read_f64_at(dtype, slice, index),
        }
    }

    /// Writes one `f32` value converted to the destination tensor data type.
    pub fn write_f32(&mut self, index: usize, val: f32) {
        match self.data {
            TensorDataMut::F32(ref mut slice) => slice[index] = val,
            TensorDataMut::F16(ref mut slice) => slice[index] = f32_to_f16(val),
            TensorDataMut::Bf16(ref mut slice) => slice[index] = f32_to_bf16(val),
            TensorDataMut::I8(ref mut slice) => {
                slice[index] = val.round_ties_even().clamp(-128.0, 127.0) as i8;
            }
            TensorDataMut::U32(ref mut slice) => {
                slice[index] = val.round_ties_even().clamp(0.0, u32::MAX as f32) as u32;
            }
            TensorDataMut::Bytes(dtype, ref mut slice) => write_f32_at(dtype, slice, index, val),
        }
    }

    /// Writes one `f64` value converted to the destination tensor data type.
    pub fn write_f64(&mut self, index: usize, val: f64) {
        self.write_f32(index, val as f32);
    }

    /// Writes raw byte at index (useful for e4m3/e5m2 and custom byte encoding).
    pub fn write_byte(&mut self, index: usize, val: u8) {
        match self.data {
            TensorDataMut::Bytes(_, ref mut slice) => slice[index] = val,
            TensorDataMut::I8(ref mut slice) => slice[index] = val as i8,
            _ => self.write_f32(index, val as f32),
        }
    }

    /// Reborrows as an immutable `TensorView`.
    pub fn as_view(&self) -> TensorView<'_> {
        let data = match self.data {
            TensorDataMut::F32(ref slice) => TensorData::F32(slice),
            TensorDataMut::F16(ref slice) => TensorData::F16(slice),
            TensorDataMut::Bf16(ref slice) => TensorData::Bf16(slice),
            TensorDataMut::I8(ref slice) => TensorData::I8(slice),
            TensorDataMut::U32(ref slice) => TensorData::U32(slice),
            TensorDataMut::Bytes(dtype, ref slice) => TensorData::Bytes(dtype, slice),
        };
        TensorView {
            shape: self.shape.clone(),
            data,
        }
    }
}

/// Owned heap-allocated multidimensional typed tensor buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedBuffer {
    shape: Vec<usize>,
    dtype: DType,
    f32_data: Vec<f32>,
    f16_data: Vec<u16>,
    bf16_data: Vec<u16>,
    i8_data: Vec<i8>,
    u32_data: Vec<u32>,
    byte_data: Vec<u8>,
}

impl TypedBuffer {
    /// Creates a zero-initialized typed buffer for the given shape and dtype.
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let total_elements: usize = shape.iter().product();
        let mut buf = Self {
            shape: shape.to_vec(),
            dtype,
            f32_data: Vec::new(),
            f16_data: Vec::new(),
            bf16_data: Vec::new(),
            i8_data: Vec::new(),
            u32_data: Vec::new(),
            byte_data: Vec::new(),
        };
        match dtype {
            DType::F32 => buf.f32_data = vec![0.0f32; total_elements],
            DType::F16 => buf.f16_data = vec![0u16; total_elements],
            DType::Bf16 => buf.bf16_data = vec![0u16; total_elements],
            DType::I8 => buf.i8_data = vec![0i8; total_elements],
            DType::U32 => buf.u32_data = vec![0u32; total_elements],
            DType::E4m3 | DType::E5m2 | DType::Bool | DType::I4 | DType::I32 => {
                let bytes = if dtype == DType::I4 {
                    total_elements.div_ceil(2)
                } else if dtype == DType::I32 {
                    total_elements * 4
                } else {
                    total_elements
                };
                buf.byte_data = vec![0u8; bytes];
            }
        }
        buf
    }

    /// Creates a buffer from `f32` slice.
    pub fn from_f32(shape: &[usize], values: &[f32]) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(values.len(), expected);
        Self {
            shape: shape.to_vec(),
            dtype: DType::F32,
            f32_data: values.to_vec(),
            f16_data: Vec::new(),
            bf16_data: Vec::new(),
            i8_data: Vec::new(),
            u32_data: Vec::new(),
            byte_data: Vec::new(),
        }
    }

    /// Creates a buffer from `f16` (u16 bits) slice.
    pub fn from_f16(shape: &[usize], values: &[u16]) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(values.len(), expected);
        Self {
            shape: shape.to_vec(),
            dtype: DType::F16,
            f32_data: Vec::new(),
            f16_data: values.to_vec(),
            bf16_data: Vec::new(),
            i8_data: Vec::new(),
            u32_data: Vec::new(),
            byte_data: Vec::new(),
        }
    }

    /// Creates a buffer from `bf16` (u16 bits) slice.
    pub fn from_bf16(shape: &[usize], values: &[u16]) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(values.len(), expected);
        Self {
            shape: shape.to_vec(),
            dtype: DType::Bf16,
            f32_data: Vec::new(),
            f16_data: Vec::new(),
            bf16_data: values.to_vec(),
            i8_data: Vec::new(),
            u32_data: Vec::new(),
            byte_data: Vec::new(),
        }
    }

    /// Creates a buffer from `i8` slice.
    pub fn from_i8(shape: &[usize], values: &[i8]) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(values.len(), expected);
        Self {
            shape: shape.to_vec(),
            dtype: DType::I8,
            f32_data: Vec::new(),
            f16_data: Vec::new(),
            bf16_data: Vec::new(),
            i8_data: values.to_vec(),
            u32_data: Vec::new(),
            byte_data: Vec::new(),
        }
    }

    /// Creates a buffer from `u32` slice.
    pub fn from_u32(shape: &[usize], values: &[u32]) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(values.len(), expected);
        Self {
            shape: shape.to_vec(),
            dtype: DType::U32,
            f32_data: Vec::new(),
            f16_data: Vec::new(),
            bf16_data: Vec::new(),
            i8_data: Vec::new(),
            u32_data: values.to_vec(),
            byte_data: Vec::new(),
        }
    }

    /// Creates an E4M3 byte buffer from raw bytes.
    pub fn from_e4m3_bytes(shape: &[usize], bytes: &[u8]) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(bytes.len(), expected);
        Self {
            shape: shape.to_vec(),
            dtype: DType::E4m3,
            f32_data: Vec::new(),
            f16_data: Vec::new(),
            bf16_data: Vec::new(),
            i8_data: Vec::new(),
            u32_data: Vec::new(),
            byte_data: bytes.to_vec(),
        }
    }

    /// Returns tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns tensor rank.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns tensor data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns total number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Reads one element converted to `f32`.
    pub fn read_f32(&self, index: usize) -> f32 {
        match self.dtype {
            DType::F32 => self.f32_data[index],
            DType::F16 => f16_to_f32(self.f16_data[index]),
            DType::Bf16 => bf16_to_f32(self.bf16_data[index]),
            DType::I8 => self.i8_data[index] as f32,
            DType::U32 => self.u32_data[index] as f32,
            DType::E4m3 => fp8_e4m3_decode(self.byte_data[index]),
            DType::E5m2 => fp8_e5m2_decode(self.byte_data[index]),
            DType::Bool | DType::I4 | DType::I32 => read_f32_at(self.dtype, &self.byte_data, index),
        }
    }

    /// Writes one `f32` value into buffer at `index`.
    pub fn write_f32(&mut self, index: usize, val: f32) {
        match self.dtype {
            DType::F32 => self.f32_data[index] = val,
            DType::F16 => self.f16_data[index] = f32_to_f16(val),
            DType::Bf16 => self.bf16_data[index] = f32_to_bf16(val),
            DType::I8 => self.i8_data[index] = val.round_ties_even().clamp(-128.0, 127.0) as i8,
            DType::U32 => {
                self.u32_data[index] = val.round_ties_even().clamp(0.0, u32::MAX as f32) as u32;
            }
            DType::E4m3 => self.byte_data[index] = fp8_e4m3_encode(val),
            DType::E5m2 => self.byte_data[index] = fp8_e5m2_encode(val),
            DType::Bool | DType::I4 | DType::I32 => {
                write_f32_at(self.dtype, &mut self.byte_data, index, val);
            }
        }
    }

    /// Reads raw byte at index for FP8/byte dtypes.
    pub fn read_byte(&self, index: usize) -> u8 {
        match self.dtype {
            DType::E4m3 | DType::E5m2 | DType::Bool | DType::I4 | DType::I32 => {
                self.byte_data[index]
            }
            DType::I8 => self.i8_data[index] as u8,
            _ => self.read_f32(index) as u8,
        }
    }

    /// Writes raw byte at index for FP8/byte dtypes.
    pub fn write_byte(&mut self, index: usize, val: u8) {
        match self.dtype {
            DType::E4m3 | DType::E5m2 | DType::Bool | DType::I4 | DType::I32 => {
                self.byte_data[index] = val;
            }
            DType::I8 => self.i8_data[index] = val as i8,
            _ => self.write_f32(index, val as f32),
        }
    }

    /// Copies data out as a vector of `f32`.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        (0..self.num_elements()).map(|i| self.read_f32(i)).collect()
    }

    /// Copies data out as a vector of `i8`.
    pub fn to_i8_vec(&self) -> Vec<i8> {
        match self.dtype {
            DType::I8 => self.i8_data.clone(),
            _ => (0..self.num_elements())
                .map(|i| self.read_f32(i) as i8)
                .collect(),
        }
    }

    /// Copies data out as a vector of raw bytes.
    pub fn to_byte_vec(&self) -> Vec<u8> {
        match self.dtype {
            DType::E4m3 | DType::E5m2 | DType::Bool | DType::I4 | DType::I32 => {
                self.byte_data.clone()
            }
            DType::I8 => self.i8_data.iter().map(|&x| x as u8).collect(),
            _ => (0..self.num_elements())
                .map(|i| self.read_f32(i) as u8)
                .collect(),
        }
    }

    /// Borrows buffer as an immutable `TensorView`.
    pub fn as_view(&self) -> TensorView<'_> {
        let data = match self.dtype {
            DType::F32 => TensorData::F32(&self.f32_data),
            DType::F16 => TensorData::F16(&self.f16_data),
            DType::Bf16 => TensorData::Bf16(&self.bf16_data),
            DType::I8 => TensorData::I8(&self.i8_data),
            DType::U32 => TensorData::U32(&self.u32_data),
            DType::E4m3 | DType::E5m2 | DType::Bool | DType::I4 | DType::I32 => {
                TensorData::Bytes(self.dtype, &self.byte_data)
            }
        };
        TensorView {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Borrows buffer as a mutable `TensorViewMut`.
    pub fn as_view_mut(&mut self) -> TensorViewMut<'_> {
        let dtype = self.dtype;
        let data = match dtype {
            DType::F32 => TensorDataMut::F32(&mut self.f32_data),
            DType::F16 => TensorDataMut::F16(&mut self.f16_data),
            DType::Bf16 => TensorDataMut::Bf16(&mut self.bf16_data),
            DType::I8 => TensorDataMut::I8(&mut self.i8_data),
            DType::U32 => TensorDataMut::U32(&mut self.u32_data),
            DType::E4m3 | DType::E5m2 | DType::Bool | DType::I4 | DType::I32 => {
                TensorDataMut::Bytes(dtype, &mut self.byte_data)
            }
        };
        TensorViewMut {
            shape: self.shape.clone(),
            data,
        }
    }
}
