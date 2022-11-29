use std::vec;
use ndarray::{Array, Dim, arr1, Axis};
use rand::seq::SliceRandom;

use super::types::{ArrT, ArrT1};

// SOFTMAX W2V
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

// NCE W2V

pub fn unigram_dist(count_data: &[String]) -> Vec<i32>{
    let count_data_iter = count_data.clone().into_iter();
    let mut w_freq = vec![0; count_data.len()];
    for (ind, word) in count_data_iter.enumerate() {
        w_freq[ind] = count_data_iter.filter(|w| *w==word).count() as i32;
    }
    w_freq
}

pub fn normalize(u_dist: &[i32]) -> Vec<f32> {
    let u_dist_iter = u_dist.clone().into_iter();
    
    let dist_max = *u_dist_iter.max().unwrap() as f32;
    let dist_min = *u_dist_iter.min().unwrap() as f32;
    u_dist_iter.map(|val| (*val as f32 -dist_min)/(dist_max-dist_min)).collect()
}

pub fn select_k_neg(n_dist: &[f32], mid_wrd_ctx: &[i32], k: &usize) -> Vec<i32> {
    let mut buf: Vec<i32> = vec![0; *k];
    for (rnd, slot) in n_dist.choose_multiple(&mut rand::thread_rng(), *k).zip(buf.iter_mut()) {
        
        *slot = n_dist.iter().position(|v| v==rnd).unwrap() as i32;
        while buf.iter().filter(|ind| *ind==slot).count() > 1 || mid_wrd_ctx.contains(slot) {
            let rnd_ = n_dist.choose(&mut rand::thread_rng()).unwrap();
            *slot = n_dist.iter().position(|v| v==rnd_).unwrap() as i32;
        }
    }
    buf
}

pub fn sigmoid(orig: &mut ArrT1){
    orig.map(|pred| *pred = 1. / (1. + (1. / pred.exp() ) ));
}

pub fn nce_log_probability(hidden: &ArrT1, opt_matrix: &ArrT) -> f64{
    let mut result = 0.;
    for ctx in opt_matrix.axis_iter(Axis(0)) {
        result += {
            let rs = hidden.dot(&ctx);
            let sg = 1. / (1. + (1. / rs.exp()) );
            sg.ln()
        }
    }
    result
}