use core::panic;
use std::vec;
use itertools::Itertools;
use ndarray::{Array, Dim, arr1, Axis};
use rand::{Rng, seq::SliceRandom};

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

pub fn noise(count_data: &[String], groomed_data: &[String]) -> Vec<f32> {
    let w_dist = unigram_dist(count_data, groomed_data);
    let alpha = 3./4.;
    let norm_wdist = normalize(&w_dist);
    println!("Normalized dist: {:?}", norm_wdist.iter().sum::<f32>());
    norm_wdist.iter().map(|w| f32::powf(*w as f32, alpha)).collect()
} 

pub fn unigram_dist(count_data: &[String], groomed_data: &[String]) -> Vec<i32>{
    let mut w_freq = vec![0; groomed_data.len()];
    for (ind, word) in groomed_data.iter().enumerate() {
        w_freq[ind] = count_data.iter().filter(|w| *w==word).count() as i32;
    }
    w_freq
}

pub fn normalize(u_dist: &[i32]) -> Vec<f32> {
    let total = u_dist.iter().sum::<i32>();
    u_dist.iter().map(|val| *val as f32/(total as f32)).collect()
}

pub fn select_k_neg(n_dist: &[f32], mid_wrd_ctx: &[i32], k: &usize) -> Vec<i32> {
    // generar tabla lo suficientemente grande para que quepan todos los elementos originales n_dist[i]*tam_tabla
    // los mas frecuentes apareceran mas veces
    let max_table_size = 100_000;
    let scale_factor = n_dist.iter().sum::<f32>();
    let mut table: Vec<i32> = vec![0];
    for (idx, prob) in n_dist.iter().enumerate() {
        let tmp_v_size = ((*prob/scale_factor)*max_table_size as f32) as usize;
        if table.len()+tmp_v_size > max_table_size {
            panic!("max_table_size reached");
        }
        let mut tmp = vec![idx as i32; tmp_v_size];
        table.append(&mut tmp);
    }
    let selected= table.choose_multiple_weighted(&mut rand::thread_rng(), *k, |item| *item).unwrap()
            .map(|x| x.to_owned())
            .collect::<Vec<_>>();
    
    // generar numero aleatorio entre 0 y tam_tabla y seleccionar el elemento i-ésimo que representa
    // el indice de la palabra que se ha elegido

    // clone the array so that it isn't moved out by implicit into_iter() and can be used later.
    
    for selection in &mut selected.clone() {
        while selected.iter().filter(|x| **x==*selection).count() > 1 || mid_wrd_ctx.contains(selection) {
            let ind_selection = rand::thread_rng().gen_range(0..table.len());
            *selection = table[ind_selection];
        }
    }
    selected
}

pub fn sigmoid(orig: &mut ArrT1){
    orig.map_mut(|pred| *pred = 1. / (1. + (1. / pred.exp() ) ));
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