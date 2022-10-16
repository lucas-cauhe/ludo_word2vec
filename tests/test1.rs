// I've gotta test: Training time for different datasets and MODEL properties (nº of neurons in hidden layer, dropout, w_size, etc..), Analogical addition, 
// Performance when applied to some NN, acquaracy (log error)
// 

#[cfg(test)]
mod tests {
    use word_embeddings::{CustomActivationFunction, CustomProbFunctionType, SkipGram, HyperParamsTune, HyperParams};
    use ndarray::{arr1};
    use serde::{Deserialize, Deserializer, Serialize, de::{self, SeqAccess}};
    use std::{marker::PhantomData, thread::JoinHandle, sync::{Arc, Mutex}, collections::HashMap};
    use std::{fmt, thread};
    
    const TRY_OUTS: usize = 4;

    #[derive(Deserialize, Serialize)]
    struct Writer {
        #[serde(deserialize_with = "deserialize_weights")]
        input_weights: Vec<Vec<f64>>,
        #[serde(deserialize_with = "deserialize_weights")]
        output_weights: Vec<Vec<f64>>
    }

    fn deserialize_weights<'de, D>(deserializer: D) -> Result<Vec<Vec<f64>>, D::Error>
    where 
        D: Deserializer<'de> 
    {
        struct PrimitiveToVec(PhantomData<Vec<Vec<f64>>>);

        impl<'de> de::Visitor<'de> for PrimitiveToVec {
            type Value = Vec<Vec<f64>>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("Matrix of 64-bit floating numbers")
            }

            fn visit_seq<A>(self, seq: A) -> Result<Vec<Vec<f64>>, A::Error>
                where
                    A: SeqAccess<'de>, {
                Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
            }
        }
        let visitor = PrimitiveToVec(PhantomData);
        deserializer.deserialize_seq(visitor)
    }


    
    static mut MODEL: SkipGram = SkipGram {
        w_size: 3,
        d: 80,
        lr: 0.001,
        prob_function: CustomProbFunctionType::Softmax,
        activation_fn: CustomActivationFunction::Sigmoid,
        batches: 20,
        train_split: 0.85,
        epochs: 1,
        data: None,
        k: None,
    };


    // currently applying leave-P-out X-validation
    #[test]
    fn a_dev_training() {

        // for some iterations -> copy MODEL and build a new model from HyperParamsTune

        let mut tuner = HyperParamsTune::new().initialize();
        let mut tuning_params: Vec<HyperParams> = vec![Default::default(); TRY_OUTS];
        for i in 0..TRY_OUTS {
            tuning_params[i] = tuner.new_hyperparams();
        }
        // It'll store the final avg error for i-th training sample
        let results = Arc::new(Mutex::new(vec![0.; TRY_OUTS])); 
        let ctx_map: HashMap<i32, Vec<i32>>;
        unsafe {
            ctx_map = MODEL.preprocess_data("/Users/cinderella/Documents/word-embeddings/tests/word_set.txt", false).unwrap();
        }
        

        let mut exec_threads = 0;
        while exec_threads < TRY_OUTS {
            let mut pool : Vec<JoinHandle<_>> = Vec::<_>::with_capacity(4);
            for _ in 0..4{
                let mut model_clone: SkipGram;
                unsafe { model_clone = MODEL.clone();}

                let results_clone = Arc::clone(&results);
                let curr_thread = exec_threads.clone();
                let tuning_model = tuning_params[exec_threads].clone();
                let ctx_map_clone = ctx_map.clone();

                let handle = thread::spawn(move || {
                    println!("Starting thread {curr_thread}");
                    let data_len = model_clone.data.as_ref().unwrap().len();
                    model_clone.d = ( data_len as f64 * tuning_model.hidden_units) as i32;
                    model_clone.lr = tuning_model.lr;
                    model_clone.batches = tuning_model.batch_size;
        
                    let (_trained_weights, avg_error) = model_clone.train(&ctx_map_clone).expect("Smth went wrong");
                    let mut res_locked = results_clone.lock().unwrap();
                    let res_element = res_locked.get_mut(curr_thread).unwrap();
                    *res_element = avg_error;
                    
                });
                pool.push(handle);
                exec_threads += 1;
            }
            for handle in pool {
                handle.join().expect("Error joining thread");
            }
        }   
        // since Ord isn't implemented for f64 you can't use fn min on Iter
        // other option would be implementing a custom nonNaN f64 type to work with the whole project (see https://stackoverflow.com/questions/28247990/how-to-do-a-binary-search-on-a-vec-of-floats/28248065#28248065 )
        
        let result_locked = results.lock().unwrap();
        let best_result = result_locked.iter().fold(0./0., |min, new| f64::max(min, *new));
        let best_params = {
            let tmp = result_locked.iter().position(|r| *r == best_result).unwrap();
            tuning_params.get(tmp).unwrap().to_owned()
        };
        println!("Best suited Hyper Params: \n{:?}", best_params);
        println!("With resulting error: {best_result}");
        println!("Out of errors: {:?}", result_locked);

        // set MODEL with best hyper params

        unsafe {
            MODEL.batches = best_params.batch_size;
            let data = MODEL.data.clone();
            MODEL.d = (data.unwrap().len() as f64 * best_params.hidden_units) as i32;
            MODEL.lr = best_params.lr;
        }
    
    }
    
    /* #[test]
    fn b_training() {
        let mut trained_weights: Vec<T> = Default::default();
        let mut avg_error: f64 = 0.;
        unsafe {
            MODEL.epochs = 20;
            let ctx_map = MODEL.preprocess_data("/Users/cinderella/Documents/word-embeddings/tests/word_set.txt", false).unwrap(); // in case its needed
            let results = MODEL.train(&ctx_map).expect("Smth went wrong");
            trained_weights = results.0;
            avg_error = results.1;
            
        }
        let to_write = Writer {
            input_weights: trained_weights[0].rows().into_iter().map(|r| r.to_vec()).collect(),
            output_weights: trained_weights[1].rows().into_iter().map(|r| r.to_vec()).collect()
        };
        std::fs::write("/Users/cinderella/Documents/word-embeddings/tests/trained_weights.json",
        serde_json::to_string_pretty(&to_write).unwrap()).unwrap();
        println!("Resulting overall_error: {avg_error}");
    } */


    #[test]
    fn c_prediction() {
        let inputs = vec!["spain", "madrid", "germany"];
    
        
        let to_read = {
            let text = std::fs::read_to_string("/Users/cinderella/Documents/word-embeddings/tests/trained_weights.json").unwrap();
            serde_json::from_str::<Writer>(&text).expect("BAAAD")
        };
        let in_shape = (to_read.input_weights.len(), to_read.input_weights[0].len());
        let out_shape = (to_read.output_weights.len(), to_read.output_weights[0].len());
        let in_flatten = to_read.input_weights.iter().flat_map(|r| r.to_owned()).collect::<Vec<f64>>();
        let out_flatten = to_read.output_weights.iter().flat_map(|r| r.to_owned()).collect::<Vec<f64>>();
        let w_in = arr1(&in_flatten).into_shape(in_shape).unwrap();
        let w_out = arr1(&out_flatten).into_shape(out_shape).unwrap();
        /* println!("input_weights: {:?}", w_in);
        println!("output_weights: {:?}", w_out); */
        let _r = unsafe {MODEL.predict(&w_in, &w_out, &MODEL, &inputs).expect("Error while training")};
    
    }
}

