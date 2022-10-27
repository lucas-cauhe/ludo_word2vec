/* 
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

} */