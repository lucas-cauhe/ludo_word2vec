/* 
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
} */