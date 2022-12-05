use std::vec;
use ndarray::{Array, Dim, arr1, Axis};
use rand::Rng;

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
    let mut w_freq = vec![0; count_data.len()];
    for (ind, word) in count_data.iter().enumerate() {
        w_freq[ind] = count_data.iter().filter(|w| *w==word).count() as i32;
    }
    w_freq
}

pub fn normalize(u_dist: &[i32]) -> Vec<f32> {
    // let u_dist_iter = u_dist.clone().into_iter();
    
    let dist_max = *u_dist.iter().max().unwrap() as f32;
    let dist_min = *u_dist.iter().min().unwrap() as f32;
    u_dist.iter().map(|val| ((*val as f32 -dist_min)/(dist_max-dist_min)).powf(0.75)).collect()
}

pub fn select_k_neg(n_dist: &[f32], mid_wrd_ctx: &[i32], k: &usize) -> Vec<i32> {
    // generar tabla lo suficientemente grande para que quepan todos los elementos originales n_dist[i]*tam_tabla
    // los mas frecuentes apareceran mas veces
    let max_table_size = 100_000;
    let mut table: Vec<i32> = vec![0; max_table_size];
    for (idx, prob) in n_dist.iter().enumerate() {
        let tmp_v_size = (*prob*max_table_size as f32) as usize;
        let mut tmp = vec![idx as i32; tmp_v_size];
        table.append(&mut tmp);
    }
    let mut selected = vec![0; *k];
    // generar numero aleatorio entre 0 y tam_tabla y seleccionar el elemento i-ésimo que representa
    // el indice de la palabra que se ha elegido
    let mut check_ind: usize = 0;
    while check_ind < *k {
        loop {
            let chosen = rand::thread_rng().gen_range(0..max_table_size);
            if !selected.contains(&table[chosen]) && !mid_wrd_ctx.contains(&table[chosen]) {
                selected[check_ind] = table[chosen];
                break;
            }
        }
        check_ind += 1;
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