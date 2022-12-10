extern crate ndarray;
extern crate serde_json;
extern crate ndarray_rand;
use std::{collections::{HashMap},cell::{Cell, RefCell}};

use probability_functions::implementors::basics::ProbFunction;
use ndarray_rand::rand::{self, Rng};

pub mod probability_functions;
mod utils;
use utils::types::*;

// Before calling the NN you must have handled:     Whether to lemmatize and stem your vocab (besides removing punctuation)
//                                                  Remove duplicate words

mod word2vec {
    
    // handle type implementations for SkipGram
    
}
/*
        Hyperparameter tuning -> 1st order: Learning Rate (logarithmic scale), #hidden units, batch-size and window size
                                 2nd order: learning rate decay, momentum term

        I'm aiming to a caviar-style training so every HyperParams for every individual training process will be stored into a container
        so that when a new HyperParams is launched, it isn't already been chosen
        Every HyperParams feature will be chosen randomly from HyperParamsTune properties.
    */
#[derive(Debug, Default, Clone)]
pub struct HyperParams {
    pub lr: f64,
    pub hidden_units: f64, // if more hidden layers were added, this should be a Vec<i32>
    pub batch_size: i32,
}

#[derive(Debug, Default, Clone)]
pub struct HyperParamsTune {
    lr: Vec<f64>,
    hidden_units: Vec<f64>, // if more hidden layers were added, this should be a Vec<i32>
    batch_size: Vec<i32>,
    in_use: HashMap<i32, Vec<(f64, f64)>> // keys are learning rates in use and the tuple are hidden_units and batch_size
}

impl HyperParamsTune {

    ///	_Creates new instance of HyperParamsTune_
    ///
    ///	# _Examples_
    ///
    /// * ``` let tuner: HyperParamsTune = HyperParamsTune::new();```

    pub fn new() -> HyperParamsTune {
        Default::default()
    }

    ///	_Configures random values for tuner fields_
    ///
    ///	# _Arguments_
    ///
    /// * `&self` 
    
    pub fn initialize(&self) -> HyperParamsTune {
        // believing it should be between 0.0001 and 0.01
        //use rand::Rng;
        let learning_rate = {
            let mut initial = vec![0.; 20];
            let mut rng = rand::thread_rng();
            for x in initial.iter_mut() {
                *x = 10.0_f64.powf(rng.gen_range(-4.0..-2.0));
            }
            initial
        };

        // # hidden_units could go from 1 to 2 times input size
        let hidden_units = {
            let mut initial = vec![0.; 20];
            let mut rng = rand::thread_rng();
            for x in initial.iter_mut() {
                *x = rng.gen_range(1.0..2.0);
            }
            initial
        };
        // batch_size could should either be 64 or 128 (or consecutive powers of 2) depending on the training data amount

        let batch_size = vec![32, 64, 128];

        HyperParamsTune { 
            lr: learning_rate,
            hidden_units, 
            batch_size,
            in_use: HashMap::<i32, Vec<(f64, f64)>>::new(),
        }

    } 

    fn is_used(&self, p: &HyperParams) -> bool {
        let pos = self.in_use.keys().position(|x| *x == p.batch_size);
        let exists = match pos {
            Some(_) => {
                let tup = (p.hidden_units, p.lr);
                let used_tup = self.in_use.get(&p.batch_size).unwrap();
                used_tup.contains(&tup)
            },
            None => false
        };
        exists
    }

    ///	_Retrieves random values from tuner_
    ///
    ///	# _Arguments_
    ///
    /// * `&mut self`
    /// 
    /// # _Examples_
    /// 
    /// ```
    /// let tuner = HyperParams::new().initialize();
    /// let tuning_params = tuner.new_hyperparams(); 
    /// ```

    pub fn new_hyperparams(&mut self) -> HyperParams {
        let mut rng = rand::thread_rng();
        let mut ret_params: HyperParams;
        loop {
            ret_params = HyperParams { 
                lr: *self.lr.get(rng.gen_range(0..20)).unwrap(), 
                hidden_units: *self.hidden_units.get(rng.gen_range(0..20)).unwrap(), 
                batch_size: self.batch_size[rng.gen_range(0..3)],
            };

            if !self.is_used(&ret_params){
                break;
            }
        }
        let entry = self.in_use.entry(ret_params.batch_size).or_insert(Vec::<(f64, f64)>::new());
        entry.push((ret_params.hidden_units, ret_params.lr));
        ret_params
    }

}

// #[derive(Clone)]
pub struct SkipGram {
    pub prob_function: Box<dyn ProbFunction>,
    pub data: Cell<Option<Vec<String>>>
}

impl SkipGram {
    
    
    ///	Builds a context map from model data
    ///
    ///	# _Arguments_
    ///
    /// * `path` - _model data location_
    /// * `for_hs` - `todo`
    /// 
    pub fn preprocess_data(&mut self, path: &str) ->  Result<HashMap<i32, Vec<i32>>, String> { 
        /* if let Some(_) = for_hs.into() {
            use Hsoftmax::{HuffmanTree};
            let processed_data = HuffmanTree::count_frequency(&d.data);
            processed_data.to_vec(); // Turn the hasmap into a vector containing tuples (<word>, <frequency>)
        } */
        let mut buffer: Vec<String> = vec![];
        let ctx_map = self.prob_function.preprocess_data(path, &mut buffer)?;
        self.data.set(Some(buffer));
        Ok(ctx_map)
        // lemmatizing, etc.. (not for now)
    }

    ///	_Trains a preprocessed model_
    /// Returns a vector, float tuple where `Vec<T>` is the trained weights vector.
    /// `f64` represents the error computed at the end of training process.
    ///
    ///	# _Arguments_
    ///
    /// * `&self`
    /// * `ctx_map` - _training data context map_
    ///
    /// # _Examples_
    /// 
    /// ```
    /// let ctx_map = model.preprocess_data("path/to/data").unwrap();
    /// let (trained_weights, overall_error) = model.train(&ctx_map).expect("Error during training: ");
    /// ```


    pub fn train(&mut self, ctx_map: &HashMap<i32, Vec<i32>>) -> Result<(Vec<ArrT>, f64), String> {
        // Here you have to obtain metrics as the model gets trained
        
        /* match self.prob_function {
            CustomProbFunctionType::Softmax => Ok(softmax::train(&self, ctx_map)?),
            CustomProbFunctionType::Hsoftmax => Err("Error".to_string()),//Hsoftmax::train(self_copy),
            CustomProbFunctionType::NCE => Ok(nce::train(&self, ctx_map)?)//NCE::train(self_copy),
        } */

        self.prob_function.train(ctx_map)

    }

    /* pub fn predict(&self, w_in: &ArrT, w_out: &ArrT, model: &SkipGram, inputs: &[&str]) -> Result<(), String> {
        // Here you have to obtain metrics as the model gets trained
        
        /* match self.prob_function {
            CustomProbFunctionType::Softmax => Ok(softmax::predict(w_in, w_out, model, inputs)),
            CustomProbFunctionType::Hsoftmax => Err("Error".to_string()),//Hsoftmax::train(self_copy),
            CustomProbFunctionType::NCE => Err("Error".to_string())//NCE::train(self_copy),
        } */
    } */
}