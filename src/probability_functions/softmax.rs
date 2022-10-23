
extern crate ndarray;
extern crate ndarray_rand;

use std::collections::HashMap;
extern crate activation_functions;
use std::ops::{Range};
use std::{vec, cmp};

use crate::SkipGram;
use crate::utils;
use utils::*;
use ndarray::{arr1};
use ndarray::{ArrayBase};

fn feed_forward(weights: &[ArrT], initial_hidden: &ArrT1, window: &[i32], ) -> Result<(ArrT1, ArrT1), String> {
    let mut next_input = initial_hidden.clone();
    let mut last_hidden: ArrT1 = arr1(&[0.; 2]);

    
    for i in 1..weights.len(){
        if i == weights.len()-1 {
            last_hidden = next_input.clone();
        }
        next_input = next_input.dot(&weights[i]);
    }
    let prediction = &softmax(&next_input);
    let ytrue = one_hot(window,  prediction.len() as i32); 
    let error_ctx = &(prediction - &arr1(&ytrue)); 
    
    Ok((((window.len()-1) as f64)*prediction + error_ctx, last_hidden))
}

// Not the standard way to go, however it works (somewhat)
/* fn standarize_hiddens(layers: &[ArrT1], denominator: f64, nn_structure: &[i32]) -> Vec<ArrT1> {
    // Should I apply Batch Norm only here ?? To the entire arch ?? Simple hidden layers avg ?? 
    // For now I'll stick with a (perhaps) non-sense hidden layers avg and normalization
    // hidden layers standarization

    let mean_hiddens: Vec<ArrT1>  = layers.iter().map(|h_layer| h_layer / denominator).collect();
    
    let standarization_avg: Vec<f64> = mean_hiddens.iter().map(|layer| layer.sum() / layer.len() as f64).collect();
    
    let standarization_stdev = {
        let mut temp: Vec<f64> = vec![0.; mean_hiddens.len()];
        for (idx, layer) in mean_hiddens.iter().enumerate() {
            temp[idx] = (layer.map(|x| (x-standarization_avg[idx]).powf(2.)).sum() / layer.len() as f64).sqrt();
        }
        temp
    };
    // standarize all hidden layers' neurons
    let std_hidden: Vec<ArrT1> = {
        let mut temp: Vec<ArrT1> = vec![arr1(vec![0.; 1].as_slice()); nn_structure.len()];
        for i in 0..nn_structure.len()-1 {
            temp[i] = arr1(vec![0.; nn_structure[i] as usize].as_slice());
        }
        for (idx, layer) in mean_hiddens.iter().enumerate() {
            temp[idx] = (layer - standarization_avg[idx]) / standarization_stdev[idx];
        }
        temp
    };
    std_hidden
} */

fn compute_gradients(nn_weights: &[ArrT], hidden_layer: &ArrT1, batch_error: &ArrT1) -> (ArrT1, ArrT) {
    // Since skipgram model is designed to work with just 1 hidden layer this process is unvectorized
    // However if more hidden layers were to be added (even there's no sense to it) subsequent mathematical operations
    // Should be performed.
    let grad_input = nn_weights[1].dot(batch_error);

    let batch_error_cp = batch_error.clone();
    let hidd_layer_cp = hidden_layer.clone();
    let sum_error_2d = batch_error_cp.into_shape((1, batch_error.len())).unwrap();
    let grad_output = hidd_layer_cp.into_shape((hidden_layer.len(), 1)).unwrap().dot(&sum_error_2d);
    
    (grad_input, grad_output)
}

/// implements model training for softmax probability function

pub fn train(props: &SkipGram, ctx_map: &HashMap<i32, Vec<i32>>) -> Result<(Vec<ArrT>, f64), String> {
    
    // initialize the random input and output weights matrix
    
    let split = match props.data {
        Some(ref v) => Some(v.split_at((v.len() as f32*props.train_split) as usize)),
        None => {
            None
        },
    };

    let real_length = split.unwrap().0.len() + split.unwrap().1.len();

    let d_len: i32 = split.unwrap().0.len().try_into().expect("Couldn't perform conversion to integer");
    
    
    let nn_structure = vec![props.d, real_length as i32];
    
    
    let mut network_weights  = initialize_weight_matrices(&nn_structure, real_length as i32).expect("Error initializing matrices");
    
    let mut epochs = 0;
    let mut overall_error = vec![0.; props.epochs];
    let mut test_errors = vec![0.; props.epochs];

    println!("Training with model params: {:?}", (props.batches, props.d, props.lr));

    loop { // epochs
        println!("----------EPOCH {epochs} ------------------");
        
        if epochs == props.epochs {
            break;
        }
        let mut precise_error = 0.;
        
        let batch_capacity = props.batches;
        let mut prev_batch = 0;
        let mut next_batch = cmp::min(prev_batch+batch_capacity, d_len);
        
        while prev_batch < d_len {
            println!("Starting batch {prev_batch} ");

            // exponentially weighted averages
            let mut input_gradient_average = arr1(vec![0.; props.d as usize].as_slice());
            let shape_ind = nn_structure.len()-2;
            let mut output_gradient_average: ArrT = ArrayBase::zeros((nn_structure[shape_ind] as usize, nn_structure[shape_ind+1] as usize));
            for i in prev_batch..next_batch {
                
                // perform feed-forward
                let first_hidden = network_weights[0].row(i as usize).to_owned();
                
                let (opt_error, last_hidden) = feed_forward(&network_weights.as_slice(), 
                    &first_hidden,
                    ctx_map.get(&i).unwrap()).expect("Error while feed-forward");
                
                if  f64::is_nan(opt_error.iter().sum()) {
                    println!("I am NaN");
                    println!("sum_error{:?}", &opt_error);
                    println!("For current word: {i}");
                    return Ok((network_weights, precise_error))
                }
                precise_error += log_probability(&i, ctx_map.get(&i).unwrap(), &network_weights[network_weights.len()-1], &last_hidden, d_len as usize);
                
                let (g_input, g_output) = compute_gradients(&network_weights, &last_hidden, &opt_error);
                input_gradient_average = props.beta*input_gradient_average + (1.-props.beta)*&g_input;
                output_gradient_average = props.beta*output_gradient_average + (1.-props.beta)*&g_output;
            }  

            // final step to gradient step
            network_weights[1] -= &(props.lr * output_gradient_average);
            let input_grad = props.lr * input_gradient_average;
            for i in prev_batch..next_batch {
                let mut input_gradient_row = network_weights[0].row_mut(i as usize);
                input_gradient_row -= &input_grad;
            }
            

            prev_batch = next_batch+1;
            next_batch = cmp::min(prev_batch+batch_capacity, d_len);
            
            println!("Precise error: {:?}", precise_error);
            overall_error[epochs] += precise_error;
            precise_error = 0.;
        }
        // Perform test
        let test_range = Range {
            start: split.unwrap().0.len() as i32,
            end: (split.unwrap().0.len() + split.unwrap().1.len()) as i32,
        };
        test_errors[epochs] = test(&network_weights[0], &network_weights[1], ctx_map, test_range);
        epochs += 1;
        
        
        plot(&overall_error, &test_errors);
        let input_weights = &network_weights[0];
        let i_vec = vec![input_weights.row(0), 
        input_weights.row(10), 
        input_weights.row(50), 
        input_weights.row(100)];
        println!("Current input weights sample: {:?}", i_vec);
        let output_weights = &network_weights[1];
        let o_vec = vec![output_weights.row(0), 
        output_weights.row(10), 
        output_weights.row(50), 
        output_weights.row(100)];
        println!("Current input weights sample: {:?}", o_vec);
    }
    Ok((network_weights, overall_error[overall_error.len()-1]))

}

/// implements prediction for softmax probability function

pub fn predict(w_in: &ArrT, w_out: &ArrT, model: &SkipGram, inputs: &[&str]){
    let vocab = model.data.as_ref().unwrap();
    for inp in 0..inputs.len() {
        
        if let Some(pos) = vocab.iter().position(|s| s == inputs[inp]){
            // Make prediction
            let hidden = w_in.row(pos).to_owned();
            let ypred = &hidden.dot(w_out);
            let ypred = &softmax(ypred);

            println!("Prediction for word {:?}: {:?}", inputs[inp], ypred);

        } else {
            println!("Word {:?} not found in training vocabulary", inputs[inp]);
        }
    }

}   

pub fn test(w_in: &ArrT, w_out: &ArrT, ctx_map: &HashMap<i32, Vec<i32>>, within: Range<i32>) -> f64 {
    // For each on the split range
    let mut global_error = 0.;
    for i in within {
        let hidden = w_in.row(i as usize).to_owned();
        global_error += log_probability(&i, ctx_map.get(&i).unwrap(), w_out, &hidden, w_out.ncols());
    }
    global_error
}


