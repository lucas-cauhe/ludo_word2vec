
pub mod Softmax{
    extern crate ndarray;
    extern crate ndarray_rand;
    
    use std::collections::HashMap;
    extern crate activation_functions;
    use std::ops::{Range};
    use std::{vec, cmp};
    
    use crate::SkipGram;
    use crate::utils;
    use utils::utils::*;
    use ndarray::{Array, arr1, arr2};
    use ndarray::{ArrayBase, OwnedRepr, Dim};
    use ndarray_rand::{RandomExt};
    use ndarray_rand::rand_distr::Uniform;
    
    type T = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;
    type T2 = ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>;

    enum InitializationMethods {
        Xavier,
        Default
    }

    fn initialization_method(method: InitializationMethods, dimensions: &i32) -> (f64, f64) {
        match method {
            InitializationMethods::Xavier => {
                (-(6./(*dimensions as f64)).sqrt(), (6./(*dimensions as f64)).sqrt())
            },
            InitializationMethods::Default => {
                (-(1./(2.* *dimensions as f64)), 1./(2.**dimensions as f64))
            }
        }
    }

    fn initialize_weight_matrices(nn_structure: &[i32], initial_neurons: i32) -> Result<Vec<T>, String> {

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

    fn feed_forward(weights: &[T], initial_hidden: &T2, window: &[i32], hidden_layers: &mut Vec<T2>) -> Result<(T2, T2), String> {
        let mut next_input = initial_hidden.clone();
        let mut last_hidden: T2 = arr1(&[0.; 2]);

        
        for i in 1..weights.len(){
            if i == weights.len()-1 {
                //last_hidden = Box::new(next_input.clone());
                last_hidden = next_input.clone();
            }
            let layer_i = hidden_layers.get(i).unwrap();
            hidden_layers[i] = layer_i + &next_input;
            next_input = next_input.dot(&weights[i]);
        }
        let prediction = &softmax(&next_input);
        let ytrue = OneHot(window,  prediction.len() as i32); 
        let error_ctx = &(prediction - &arr1(&ytrue)); 
        
        Ok((((window.len()-1) as f64)*prediction + error_ctx, last_hidden))
    }

    fn standarize_hiddens(layers: &[T2], denominator: f64, nn_structure: &[i32]) -> Vec<T2> {
        // Should I apply Batch Norm only here ?? To the entire arch ?? Simple hidden layers avg ?? 
        // For now I'll stick with a (perhaps) non-sense hidden layers avg and normalization
        // hidden layers standarization

        let mean_hiddens: Vec<T2>  = layers.iter().map(|h_layer| h_layer / denominator).collect();
       
        let standarization_avg: Vec<f64> = mean_hiddens.iter().map(|layer| layer.sum() / layer.len() as f64).collect();
        
        let standarization_stdev = {
            let mut temp: Vec<f64> = vec![0.; mean_hiddens.len()];
            for (idx, layer) in mean_hiddens.iter().enumerate() {
                temp[idx] = (layer.map(|x| (x-standarization_avg[idx]).powf(2.)).sum() / layer.len() as f64).sqrt();
            }
            temp
        };
        // standarize all hidden layers' neurons
        let std_hidden: Vec<T2> = {
            let mut temp: Vec<T2> = vec![arr1(vec![0.; 1].as_slice()); nn_structure.len()];
            for i in 0..nn_structure.len()-1 {
                temp[i] = arr1(vec![0.; nn_structure[i] as usize].as_slice());
            }
            for (idx, layer) in mean_hiddens.iter().enumerate() {
                temp[idx] = (layer - standarization_avg[idx]) / standarization_stdev[idx];
            }
            temp
        };
        std_hidden
    }

    fn compute_gradients(nn_weights: &[T], hidden_layers: &[T2], batch_error: &T2) -> (T2, T) {
        // Since skipgram model is designed to work with just 1 hidden layer this process is unvectorized
        // However if more hidden layers were to be added (even there's no sense to it) subsequent mathematical operations
        // Should be performed.
        let grad_input = nn_weights[1].dot(batch_error);

        let batch_error_cp = batch_error.clone();
        let hidd_layer_zero = hidden_layers.get(0).unwrap().clone();
        let sum_error_2d = batch_error_cp.into_shape((1, batch_error.len())).unwrap();
        let grad_output = hidd_layer_zero.into_shape((hidden_layers[0].len(), 1)).unwrap().dot(&sum_error_2d);
        
        (grad_input, grad_output)
    }

    pub fn train(props: &SkipGram, ctxMap: &HashMap<i32, Vec<i32>>) -> Result<Vec<T>, String> {
        
        // initialize the random input and output weights matrix
        
        let split = match props.data {
            Some(ref v) => Some(v.split_at((v.len() as f32*props.train_split) as usize)),
            None => {
                None
            },
        };

        let real_length = split.unwrap().0.len() + split.unwrap().1.len();

        let d_len: i32 = split.unwrap().0.len().try_into().expect("Couldn't perform conversion to integer");
        
        let mut precise_error = 0.;
        let nn_structure = vec![props.d, real_length as i32];
        
        
        let mut network_weights  = initialize_weight_matrices(&nn_structure, real_length as i32).expect("Error initializing matrices");
        
        let mut epochs = 0;
        let mut overall_error = vec![0.; 20];
        let mut test_errors = vec![0.; 20];

        loop { // epochs
            println!("----------EPOCH {epochs} ------------------");
            if epochs == 20 {
                break;
            }
            //batches
            let batches = 10;
            let batch_capacity = d_len / batches;
            let mut prev_batch = 0;
            let mut next_batch = cmp::min(prev_batch+batch_capacity, d_len);
            
            while prev_batch < d_len {
                println!("Starting batch {prev_batch} ");
                let mut batch_error = arr1(vec![0. as f64; real_length].as_slice());

                let mut hidden_avg: Vec<T2> = vec![arr1(vec![0.; 1].as_slice()); nn_structure.len()];
                for i in 0..nn_structure.len()-1 {
                    hidden_avg[i] = arr1(vec![0.; nn_structure[i] as usize].as_slice());
                }
                for i in prev_batch..next_batch {
                    
                    // perform feed-forward
                    let first_hidden = network_weights[0].row(i as usize).to_owned();
                    hidden_avg[0] += &first_hidden;
                    let (opt_error, last_hidden) = feed_forward(&network_weights.as_slice(), 
                        &first_hidden,
                        ctxMap.get(&i).unwrap(),
                        &mut hidden_avg).expect("Error while feed-forward");
                    
                    if  f64::is_nan(opt_error.iter().sum()) {
                        println!("I am NaN");
                        println!("sum_error{:?}", &opt_error);
                        println!("For current word: {i}");
                        return Ok(network_weights)
                    }
                    batch_error += &opt_error;
                    precise_error += log_probability(&i, ctxMap.get(&i).unwrap(), &network_weights[network_weights.len()-1], &last_hidden, d_len as usize);
                    
                    // perform gradient descent
                }  

                let std_hiddens = standarize_hiddens(&hidden_avg, (next_batch-prev_batch) as f64, &nn_structure);
                batch_error = batch_error / (next_batch-prev_batch) as f64;
                
                let (GInput, GOutput) = compute_gradients(&network_weights, &std_hiddens, &batch_error);

                // final step to gradient step
                network_weights[1] -= &(props.lr * GOutput); // .t() ?? 
                let input_grad = props.lr * GInput;
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
            test_errors[epochs] = test(&network_weights[0], &network_weights[1], ctxMap, test_range);
            epochs += 1;
            
            
            plot(&overall_error, &test_errors);
        }
        Ok(network_weights)

    }

      

    fn OneHot(on: &[i32], l: i32) -> Vec<f64> {
        let mut repr = vec![0.; l as usize];
        let true_selected: Vec<&i32> = on.iter().filter(|ind| *ind >= &0).collect(); // this will be useless when context hashmap is properly designed
        for i in true_selected {
            let val = repr.get_mut(*i as usize).unwrap();
            *val = 1.;
        }
        repr
    }

    pub fn predict(w_in: &T, w_out: &T, model: &SkipGram, inputs: &[&str]){
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

    pub fn test(w_in: &T, w_out: &T, ctxMap: &HashMap<i32, Vec<i32>>, within: Range<i32>) -> f64 {
        // For each on the split range
        let mut global_error = 0.;
        for i in within {
            let hidden = w_in.row(i as usize).to_owned();
            global_error += log_probability(&i, ctxMap.get(&i).unwrap(), w_out, &hidden, w_out.ncols());
        }
        global_error
    }

    
}

/* 
pub mod HSoftmax{
    extern crate ndarray;
    extern crate ndarray_rand;
    extern crate activation_functions;

    use crate::SkipGram;
    use activation_functions::f64::{sigmoid};
    use ndarray::{Array, arr2};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    

    type WMatrix = Vec<Vec<f64>>;
    struct TreeNode {
        id: i32,
        value: &str,
        left: Option<TreeNode>,
        right: Option<TreeNode>,
    }

    pub struct HuffmanTree {
        root: TreeNode,
        leaves: i32,
        total_nodes: i32,
    }

    impl HuffmanTree {
      
        pub fn build(&mut self, data: &Vec<(String, i32)>){
            let sorted_data = data.data.sort(); // Make this into a priority queue
            
            let tree = Self {
                root: TreeNode{id: 0, value: "", left: None, right: None},
                leaves: 0,
                total_nodes: 0,
            };

            loop {
                if (sorted_data.is_empty()) {
                    break;
                }

                let left_node = TreeNode{id: sorted_data[0].1, value: sorted_data[0].0, left: None, right: None};
                let right_node = TreeNode{id: sorted_data[1].1, value: sorted_data[1].0, left: None, right: None};
                let sum = sorted_data[0].1 + sorted_data[1].1;

                tree.root = TreeNode{id: sum, value: &sum, left: &left_node, right: &right_node};
                tree.leaves += 2;
                tree.total_nodes += 3;

                sorted_data.dequeue([0,1]);
                for d in &sorted_data {
                    if d.1 <= sum {
                        // put sum on d's place and expand de queue 1 position forward
                    }
                }

            }   
            tree

        }

        pub fn count_frequency(d_vec: &Vec<String>) -> HashMap<&str, i32, _> {
            let freq_map = HashMap::new();
            for w in d_vec{
                if !freq_map.contains_key(&w) {
                    freq_map.insert((w, 0));
                    continue
                } 
                let count = freq_map.entry(w);
                *count.1 = count.1+1;
            }
            freq_map
        }

    }

    pub fn train(data: &SkipGram) {
        // build the tree
        let tree: HuffmanTree = HuffmanTree::build_tree(&data.data);
        // traverse it to find all nodes that aren't leaves and numerate them properly (root node is id: 0 and last non-leave-node is n)
        let output_layer_dim = tree.total_nodes - tree.leaves;

        let data = match data.data {
            Some(v) => v,
            None => None
        };
        let d_len: i32 = data.len();

        let mut WInputs: WMatrix = Array::random((d_len, data.d), Uniform::new(-5., 5.));
        let mut WOutputs: WMatrix = Array::random((data.d, output_layer_dim), Uniform::new(-5., 5.));

        let mut batches = 20;
        let err = vec![0; batches];
        
        while let Some(val) = batches.checked_sub(1) {
            if val == 0{
                print!("Process exited successfully");
            }
            for (k, (w, _)) in &data.data.iter().enumerate() {
                
                let context_words = []; // get context words
                let hidden = WInputs[k];
                let mut sum_error = vec![0; &output_layer_dim];
                for ctx in context_words {
                    // Select those as the output layer for the NN
                    // During training, first you have to see where is the ctx word located in the tree and convert it to a vector
                    // If the sequence were 1 but output layer dim was maybe 6, it is interpreter job to check whether you have reached a leave node
                    // therefore, you can train the rest of the outuput layer to be whatever you want (either 0 or 1), in my case 0.
                    let ytrue = compute_ytrue(&tree.root, &ctx, &vec![0; output_layer_dim], 0);
                    
                    // Feed-forward
                    //let output_layer = MMult(&hidden, &WOutputs);
                
                    let output_layer = arr2(&hidden).dot(&WOutputs);
                    let ypred = sigmoid(output_layer);

                    let error_ctx = ypred - ytrue;
                    sum_error += error_ctx;
                }

                // perform backward propagation
                let grad_input = MMult(sum_error, transpose(&WOutputs));
                let grad_output = MMult(&hidden, &WOutputs);

                WInputs -= lr*grad_input;
                WOutputs -= lr*grad_output;


            }

            //compute log probability error
        }
        
    }

    fn compute_ytrue(t: &TreeNode, tgt: &str, mut o_layer: &Vec<i32>, node: i32) -> Option<Vec<i32>> {
        
        if t.value == tgt {
            vec![0; o_layer.length()]
        }

        if let Some(v) = compute_ytrue(&t.left, tgt, &o_layer, node+1) {
            o_layer = v; // maybe *o_layer[node] = 0; ??
            o_layer[node] = 0;
        } else {
            let v_ = compute_ytrue(&t.right, tgt, &o_layer, node+1);
            v_[node] = 1;
            o_layer = v_;
        }
        Some(o_layer)
    }


}


pub mod NCE {
    extern crate ndarray;
    extern crate ndarray_rand;

    

    use crate::SkipGram;
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use crate::utils::*;

    type WMatrix = Vec<Vec<i32>>;
    type HiddenLayer = Vec<i32>;

    pub fn train(data: &SkipGram) { // check if referenced object keeps referenced object-values (and nested values)
        
        let k = match data.k {
            Some(v) => v,
            None => 5 // set to default value k=5
        };

        let d_len: i32 = match data.data{
            Some(d) => d.len(),
            None => None,
        };
        let batches = 20;

        let mut WInputs: WMatrix = Array::random((d_len, data.d), Uniform::new(-5., 5.));
        let mut WOutputs: WMatrix = Array::random((d_len, data.d), Uniform::new(-5., 5.));


        while let Some(val) = batches.checked_sub(1) {
            if val == 0{
                print!("Process exited successfully");
            }

           
            
            // for each word in data 
            for w in [0..d_len] {
                let hidden = WInputs[w];
                // for each ctx in word context
                for ctx in get_context(data.data[w]) {
                    // compute noise words
                    let negative_pairs = PNoise(&w, &data.data);
                    // compute ytrue
                    let ytrue = vec![0; &k+1];
                    ytrue[0] = 1;

                    // perform feed forward
                    let mut ctx_weigths = Vec::from([
                        WOutputs[ctx],
                        WOutputs[negative_pairs]
                    ]);
                    let ypred = MMult(&ctx_weigths, &hidden); // (k x dim) X (dim x 1)
                    let ypred = sigmoid(ypred);

                    // compute error
                    let error = ypred-ytrue;
                    
                    // backward propagation
                    let grad_inputs = MMult(transpose(&error), &ctx_weigths); // (1 x k) X (k x dim)
                    let grad_outputs = MMult(error, &transpose(hidden)); // (k x 1) X (1 x dim)

                    WInputs[w] -= lr*grad_inputs;
                    ctx_weigths -= lr*grad_outputs;
                    WOutputs[negative_pairs] = ctx_weigths;
                }
            }
                
            // compute cost 
        }

    }

    fn PNoise() {

    }
} */
