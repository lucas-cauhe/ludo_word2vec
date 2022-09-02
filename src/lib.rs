extern crate ndarray;
use std::{collections::HashMap, borrow::{Borrow, BorrowMut}};

use itertools::Itertools;
use regex::Regex;

pub mod ProbabilityFunctions;
pub mod utils;
use utils::utils::{build_context};
use ndarray::{ArrayBase, OwnedRepr, Dim};
use ProbabilityFunctions::{Softmax};
type WMatrix = Vec<Vec<f64>>;
type T = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

// TAKE A LOOK AT THE CONTEXT STUFFx


// Before calling the NN you must have handled:     Whether to lemmatize and stem your vocab (besides removing punctuation)
//                                                  Remove duplicate words


mod word2vec {
    
    // handle type implementations for SkipGram
    
}
// This goes in nn.rs file
// Neural Network which takes as parameters: probability function (softmax, Hierarchical softmax, NCE),
//                                           window size, 
//                                           dimension of the hidden layers,
//                                           dropout (mainly applied to softmax)
//                                           activation function (ReLU, sigmoid...)
#[derive(Clone)]
pub enum CustomProbFunctionType {
    Softmax,
    HSoftmax,
    NCE
}
#[derive(Clone)]
pub enum CustomActivationFunction {
    ReLU,
    Sigmoid,
    Tanh
}

#[derive(Clone)]
pub struct SkipGram {
    pub w_size: i32, // take a look into positional windows
    pub d: i32,
    pub lr: f64,
    pub prob_function: CustomProbFunctionType,
    pub activation_fn: CustomActivationFunction,
    pub dropout: bool,
    pub batches: i32,
    pub data: Option<Vec<String>>,
    pub k: Option<i32>, //only for NCE
}

impl SkipGram {
    
    
    pub fn preprocess_data<T: Into<Option<bool>>>(&mut self, path: &str, for_hs: T) ->  Result<HashMap<i32, Vec<i32>>, String> { 
        /* if let Some(_) = for_hs.into() {
            use HSoftmax::{HuffmanTree};
            let processed_data = HuffmanTree::count_frequency(&d.data);
            processed_data.to_vec(); // Turn the hasmap into a vector containing tuples (<word>, <frequency>)
        } */
        // list all words, deleting duplicates and punctuation
        use std::fs;
        
        let content = fs::read_to_string(path).unwrap();
        let content = format!(r"{}", content);
        println!("{:?}", &content);
        let re = Regex::new(r"[[:punct:]\n]").unwrap();
        let content_ = re.replace_all(&content, "");
        
        
        println!("{:?}", &content_);

        let groomed_content: Vec<String> = content_.to_lowercase().split(' ').unique().map(|w| w.to_string()).collect();

        let context_map = build_context(&content_.to_lowercase().split(' ').collect(), &self.w_size, &groomed_content);

        
        match context_map {
            Ok(m) => {
                self.data = Some(groomed_content);
                Ok(m)
            },
            Err(_) => Err("Context map couldn't be built successfully".to_string())
        }
        
        
        // lemmatizing, etc..
    }

    pub fn train(&self, ctxMap: &HashMap<i32, Vec<i32>>) -> Result<(T, T), String> {
        // Here you have to obtain metrics as the model gets trained
        
        match self.prob_function {
            CustomProbFunctionType::Softmax => Ok(Softmax::train(&self, ctxMap)?),
            CustomProbFunctionType::HSoftmax => Err("Error".to_string()),//HSoftmax::train(self_copy),
            CustomProbFunctionType::NCE => Err("Error".to_string())//NCE::train(self_copy),
        }
    }
}