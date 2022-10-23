use ndarray::{Array, Dim, arr1};


pub fn softmax(a: &Array<f64, Dim<[usize; 1]>>) -> Array<f64, Dim<[usize; 1]>>{
    let mut tot_sum = 0.;
    let mut ret_array = arr1(&vec![0.; a.len()]);
    for v in a {
        tot_sum += v.max(1e-10).exp();
    }
    
    for (k, v) in a.iter().enumerate() {
        ret_array[k] = v.max(1e-10).exp() / tot_sum;
    }
    ret_array
}

pub fn one_hot(on: &[i32], l: i32) -> Vec<f64> {
    let mut repr = vec![0.; l as usize];
    let true_selected: Vec<&i32> = on.iter().filter(|ind| *ind >= &0).collect(); // this will be useless when context hashmap is properly designed
    for i in true_selected {
        let val = repr.get_mut(*i as usize).unwrap();
        *val = 1.;
    }
    repr
}