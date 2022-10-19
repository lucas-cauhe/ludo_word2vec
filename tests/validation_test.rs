mod dev_test {

    use word_embeddings::{CustomActivationFunction, CustomProbFunctionType, SkipGram, HyperParamsTune, HyperParams};
    use std::{thread::{self, JoinHandle}, sync::{Arc, Mutex}, collections::HashMap};

    const TRY_OUTS: usize = 20;
    static mut MODEL: SkipGram = SkipGram {
        w_size: 3,
        d: 80,
        lr: 0.001,
        prob_function: CustomProbFunctionType::Softmax,
        activation_fn: CustomActivationFunction::Sigmoid,
        batches: 20,
        train_split: 0.85,
        epochs: 10,
        beta: 0.9,
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
    
    }
}