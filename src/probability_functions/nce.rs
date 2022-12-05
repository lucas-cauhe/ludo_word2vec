extern crate ndarray;
extern crate ndarray_rand;

use std::collections::HashMap;
use std::ops::Mul;
use ndarray::{arr1, Axis};

use crate::SkipGram;
use crate::utils::funcs::{unigram_dist, select_k_neg, normalize, sigmoid, nce_log_probability};
use crate::utils::initialization::initialize_weight_matrices;
use crate::utils::types::{ArrT, ArrT1};

pub fn train(props: &SkipGram, ctx_map: &HashMap<i32, Vec<i32>>) -> Result<(Vec<ArrT>, f64), String>{ // check if referenced object keeps referenced object-values (and nested values)
    
    // initialize the random input and output weights matrix
    let split = match props.data {
        Some(ref v) => Some(v.split_at((v.len() as f32*props.train_split) as usize)),
        None => {
            None
        },
    };
    let k: usize = 10; // attending to original paper's criterion
    let real_length = split.unwrap().0.len() + split.unwrap().1.len();
    let nn_structure = vec![props.d, real_length as i32];
    let mut network_weights  = initialize_weight_matrices(&nn_structure, real_length as i32).expect("Error initializing matrices");
    
    println!("Training with model params: {:?}", (props.batches, props.d, props.lr));

    
    let mut overall_error = vec![0.; props.epochs];
    let mut test_errors = vec![0.; props.epochs];
    let d_len = split.unwrap().0.len() as i32;
    let check_weights = false;
    // define noise function
    let n_dist = noise(props.data.as_ref().unwrap());

    for epoch in 0..props.epochs {
        for word in 0..d_len {
            let hidden = network_weights[0].row(word as usize).to_owned();

            for ctx in ctx_map.get(&word).unwrap(){

                let k_neg = select_k_neg(&n_dist, ctx_map.get(&word).unwrap(), &k);
                let mut ytrue = vec![0.; k+1];
                ytrue[0] = 1.;

                // ensemble operating matrix
                let opt_matrix = ensemble_from(&k_neg, ctx, &network_weights[1]);

                // feed forward
                let ctx_err = feed_forward_nce(&hidden, &opt_matrix, &arr1(ytrue.as_slice()));

                // compute error
                let prediction_error =  nce_log_probability(&hidden, &opt_matrix);
                overall_error[epoch] += prediction_error;

                // backprop
                let gradients = nce_compute_gradients(&ctx_err, &opt_matrix, &hidden);
                update_weights(&mut network_weights, &gradients, &props.lr, &k_neg, &(word as usize))?;

            }
        }
        println!("Error computed in epoch {epoch}: {:?}", overall_error);
    }
    Ok((network_weights, overall_error[props.epochs-1]))

}

fn noise(count_data: &[String]) -> Vec<f32> {
    let w_dist = unigram_dist(count_data);
    let alpha = 3./4.;
    let norm_wdist = normalize(&w_dist);
    norm_wdist.iter().map(|w| f32::powf(*w as f32, alpha)).collect()
} 

fn ensemble_from(k_neg: &[i32], ctx: &i32, from: &ArrT) -> ArrT {
    let mut collapsed = k_neg.clone().to_vec();
    collapsed.push(*ctx);
    let u_collapsed: Vec<usize> = collapsed.iter().map(|e| *e as usize).collect();
    from.select(Axis(0), &u_collapsed)
}

fn feed_forward_nce(hidden: &ArrT1, opt_matrix: &ArrT, ytrue: &ArrT1) -> ArrT1{

    let mut prediction = opt_matrix.dot(hidden);
    sigmoid(&mut prediction);
    let error_ctx = prediction - ytrue;
    error_ctx

}

fn nce_compute_gradients(ctx_err: &ArrT1, opt_matrix: &ArrT, hidden: &ArrT1) -> (ArrT1, ArrT) {

    let grad_input = ctx_err.dot(opt_matrix);

    let ctx_err_2d = ctx_err.clone().into_shape((1, opt_matrix.nrows())).unwrap();
    let grad_output = hidden.clone().into_shape((hidden.len(), 1)).unwrap().dot(&ctx_err_2d);
    (grad_input, grad_output)
}

fn update_weights(net_weights: &mut [ArrT], grads: &(ArrT1, ArrT), lr: &f64, negs: &[i32], center_w: &usize) -> Result<(), String> {

    let mut input_weight = net_weights[0].row_mut(*center_w);
    input_weight -= &(grads.0.clone().mul(*lr));

    
    let mut err = false;
    net_weights[1].axis_iter_mut(Axis(0)).enumerate()
        .filter(|(i, _)| negs.contains(&(*i as i32)))
        .map(|(_, mut val)| val -= &(grads.1.row( {
            static mut curr_ind: usize = 0;
            unsafe {curr_ind += 1; 
            err = curr_ind > negs.len()-1; 
            curr_ind }
        }).mul(*lr)));
    
    if err {
        return Err("Error actualizando weights".to_string());
    }
    Ok(())
}