
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use plotters::prelude::*;

use std::{collections::HashMap, cmp::{min}};

use ndarray::{ArrayBase, OwnedRepr, Dim, Array, arr1, arr2};

pub type ArrT = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;
pub type ArrT1 = ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>;

pub fn build_context(c: &Vec<&str>, w_size: &i32, content_array: &Vec<String>) -> Result<HashMap<i32, Vec<i32>>, String> {
    let mut context_map = HashMap::<i32, Vec<i32>>::new(); // Key: index in cleaned dataset, Values: context indices
    
    println!("Building context map, this could take some time...");
    for w_ind in 0..c.len() {
        let checked_sub = match w_ind.checked_sub(*w_size as usize) {
            Some(s) => s,
            None => 0,
        };
        let ctx_range = checked_sub..min(c.len(), w_ind+(*w_size as usize));
        let mid_word_ind = content_array.iter().position(|w| *w==c[w_ind]).unwrap() as i32;
        for ctx_ind in ctx_range {
            if ctx_ind == w_ind {
                continue;
            }
            let ctx_word = content_array.iter().position(|w| *w==c[ctx_ind]).unwrap() as i32;
            let ctx_entry = context_map.entry(ctx_word).or_insert(Vec::new());
            ctx_entry.push(mid_word_ind);
        }
    }
    println!("Context map built successfully");
    Ok(context_map)
}

pub fn log_probability(ckey: &i32, w_indow: &[i32], w_output: &ArrT, hidden: &ArrT1, curr_len: usize) -> f64 {
    let mut brute_cost = 0.;
    
    for ctxkey in w_indow {
        let prob = prob_function(*ckey as usize, ctxkey, w_output, curr_len, hidden);
        brute_cost += f64::ln(prob);
    }
    brute_cost / curr_len as f64

}

fn prob_function(c_word: usize, ctx_word: &i32, w_output: &ArrT, d_len: usize, hidden: &ArrT1) -> f64 {
    let mut prob_sum = 0.;
    for w in 0..d_len{
        if w != c_word {
            prob_sum += f64::exp(w_output.column(w).dot(hidden));
        }
    }
    let ctx_product = f64::exp(w_output.column(*ctx_word as usize).dot(hidden));
    ctx_product / prob_sum
} 

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

pub fn plot(train_errs: &[f64], test_errs: &[f64]) {
    
    let root_area = BitMapBackend::new("/Users/cinderella/Documents/word-embeddings/data/plot.png", (600, 400)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();
    
    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 100.0)
        .set_label_area_size(LabelAreaPosition::Bottom, 40.0)
        .set_label_area_size(LabelAreaPosition::Right, 100.0)
        .set_label_area_size(LabelAreaPosition::Top, 40.0)
        .caption("Training Plot", ("sans-serif", 40.0))
        .build_cartesian_2d(0..20, 0.0..4000.0)
        .unwrap();
    ctx.configure_mesh().draw().unwrap();

    ctx.draw_series(
        LineSeries::new((0..).zip(train_errs.iter()).map(|(idx, y)| {(idx, -y)}),&BLUE)
    ).unwrap();
    ctx.draw_series(
        LineSeries::new((0..).zip(test_errs.iter()).map(|(idx, y)| {(idx, -y)}),&RED)
    ).unwrap();
}
enum InitializationMethods {
    Xavier,
    Standard,
}

fn initialization_method(method: InitializationMethods, dimensions: &i32) -> (f64, f64) {
    match method {
        InitializationMethods::Xavier => {
            (-(6./(*dimensions as f64)).sqrt(), (6./(*dimensions as f64)).sqrt())
        },
        InitializationMethods::Standard => {
            (-(1./(2.* *dimensions as f64)), 1./(2.**dimensions as f64))
        }
    }
}

pub fn initialize_weight_matrices(nn_structure: &[i32], initial_neurons: i32) -> Result<Vec<ArrT>, String> {

    let mut initialized_matrices = vec![arr2(&[[0.], [0.]]); nn_structure.len()];
    let init = initialization_method(InitializationMethods::Xavier, &initial_neurons);

    for layer in nn_structure.iter().enumerate() {
        
        if layer.0 == 0 {
            initialized_matrices[0] = Array::random((initial_neurons as usize, *layer.1 as usize), Uniform::<f64>::new(&init.0, &init.1)); 
        }else {
            initialized_matrices[layer.0] = Array::random((nn_structure[layer.0-1] as usize, *layer.1 as usize), Uniform::<f64>::new(&init.0, &init.1)); 
        }
    }
    Ok(initialized_matrices)
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