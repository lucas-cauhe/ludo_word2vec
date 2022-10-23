use ndarray::{ArrayBase, OwnedRepr, Dim};

pub type ArrT = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;
pub type ArrT1 = ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>;