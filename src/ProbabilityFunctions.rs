
pub mod Softmax{
    extern crate ndarray;
    extern crate ndarray_rand;
    use std::collections::HashMap;
    extern crate activation_functions;
    use std::ops::Sub;
    use std::{vec,};
    
    use crate::SkipGram;
    use crate::utils;
    use utils::utils::*;
    use ndarray::{Array, arr1};
    use ndarray::{ArrayBase, OwnedRepr, Dim};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    
    type T = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;

    pub fn train<'a>(props: &SkipGram, ctxMap: &HashMap<i32, Vec<i32>>) -> Result<(T, T), String> {
        
        // initialize the random input and output weights matrix
        let non: Vec<String> = Vec::new();
        let data = match props.data {
            Some(ref v) => v,
            None => {
                &non
            },
        };
        let d_len: i32 = data.len().try_into().expect("Couldn't perform conversion to integer");
        let mut prev_error = -1000000.;
        let mut precise_error = 0.;
        let six: f64 = 6.;
        let init = (-(six/(d_len as f64)).sqrt(), (six/(d_len as f64)).sqrt()); // Xavier initialization
        let mut WInputs = Array::random((d_len as usize, props.d as usize), Uniform::<f64>::new(&init.0, &init.1)); 
        let mut WOutputs = Array::random((props.d as usize, d_len as usize), Uniform::<f64>::new(&init.0, &init.1));

        
        
        // for a number (preset) of iterations or until error converges to 0
        let mut val = props.batches;
        loop { // start with precise error
            
            
            //println!("Batch: {} is starting now", val);
            // for each word in data
            for i in 0..d_len {
                //println!("{i}/{d_len}");
                // perform feed-forward
                
                let hidden = WInputs.row(i.try_into().unwrap()).to_owned();
                // if activation function is taken into account here should be utilized

                let ypred = &hidden.dot(&WOutputs);
                
                //apply softmax
                let ypred = &softmax(ypred);
                // take context words one-hot representation ( [0 0 1 1 0 1 1 0 0 0]  = ytrue)
                let ytrue = &OneHot(ctxMap.get(&i).unwrap(), d_len);
                
                // compute error from the output layer -> ypred - ytrue
                // compute the sum error like: (k-1)*ypred + error_ctx
                let error_ctx = ypred - &arr1(ytrue); 
                // sum_error should eventually converge to 0
                let sum_error = ((ctxMap.get(&i).unwrap().len()-1) as f64)*ypred + error_ctx;
                
                // for each computed sum error, add it to acc_error and decide which performs better (log_prob or this one)
                // idealy you should choose between precise/thorough loss function (log_prob) or ypred sum one
                
                let quick_error = sum_error.iter().map(|x| x.abs()).sum::<f64>(); // the total error is both x axis sided
                if quick_error < 10. {
                    println!("Current sum_error for indexed word {}, is : {:?}", &i, quick_error);
                }
                precise_error += log_probability(&i, ctxMap, ypred);
                // perform gradient descent
                
                let grad_input = WOutputs.dot(&sum_error);
                let s_len = sum_error.len();
                let dim = props.d as usize;
                let sum_error_2d = sum_error.into_shape((s_len, 1)).unwrap();
                let hidden_2d = hidden.into_shape((1, dim)).unwrap();
                let grad_output = sum_error_2d.dot(&hidden_2d);
                

                let prod = props.lr*grad_input;
                
                let mut ith_row = WInputs.row_mut(i as usize);
                {
                    let ith_row_cp = ith_row.to_owned();
                    let subt  = ith_row_cp - prod;
                    
                    ith_row.assign(&subt);
                }
                
                
                let out_prod = (props.lr*grad_output);
                WOutputs = WOutputs.sub(&out_prod.t());
                //println!("Current output prod: {:?}", out_prod);

                if  f64::is_nan(out_prod[(0,0)]) {
                    println!("I am NaN");
                    println!("sum_error{:?}", sum_error_2d);
                    return Ok((WInputs, WOutputs))
                }

            }
            precise_error = precise_error / d_len as f64;
            println!("Precise error: {:?}", precise_error);

            
            
            if prev_error > precise_error+5. {
                break;
            }
            prev_error = precise_error;
            precise_error = 0.;
            
            
        }
        Ok((WInputs, WOutputs))

    }

      

    fn OneHot(on: &Vec<i32>, l: i32) -> Vec<f64> {
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
