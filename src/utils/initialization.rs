use ndarray::{arr2, Array};
use ndarray_rand::rand_distr::Uniform;

use super::enums::InitializationMethods;
use super::types::*;

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