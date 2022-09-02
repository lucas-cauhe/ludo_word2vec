

pub mod utils {
    use std::{collections::HashMap, cmp::min};

    use ndarray::{ArrayBase, OwnedRepr, Dim, Array, arr1};
    type T2 = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;
    type T1 = ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>;
    /* fn MMult(m1: &WMatrix, m2: &WMatrix) -> Result<WMatrix, io::Error> {
        // If you see any crate which does matrix multiplication, implement it for WMatrix type
        // otherwise, simply multiply 🙃
        
    } */
    
    pub fn build_context(c: &Vec<&str>, w_size: &i32, content_array: &Vec<String>) -> Result<HashMap<i32, Vec<i32>>, String> {
        let mut context_map = HashMap::<i32, Vec<i32>>::new(); // Key: index in cleaned dataset, Values: context indices
        
        println!("Building context map, this could take some time...");
        for wInd in 0..c.len() {
            let checked_sub = match wInd.checked_sub(*w_size as usize) {
                Some(s) => s,
                None => 0,
            };
            let ctxRange = checked_sub..min(c.len()-1, wInd+(*w_size as usize));
            let midWordInd = content_array.iter().position(|w| *w==c[wInd]).unwrap() as i32;
            for ctxInd in ctxRange {
                if ctxInd == wInd {
                    continue;
                }
                let ctxWord = content_array.iter().position(|w| *w==c[ctxInd]).unwrap() as i32;
                let ctxEntry = context_map.entry(ctxWord).or_insert(Vec::new());
                ctxEntry.push(midWordInd);
            }
        }
        println!("Context map built successfully");
        Ok(context_map)
    }
    
    pub fn log_probability(T: &i32, k: &i32, WOutput: &T2, ckey: &usize, h: &T1, ctxMap: &HashMap<i32, Vec<i32>>) -> f64 {
        let mut brute_cost = 0.;
        
        for ctxkey in ctxMap.get(k).unwrap() {
            let prob = prob_function(*ckey, ctxkey, WOutput, h);
            brute_cost += f64::ln(prob);
        }
        brute_cost

    }

    fn prob_function(c_word: usize, ctx_word: &i32, non_ctx_words: &T2, h: &T1) -> f64 {
        let mut prob_sum = 0.;
        for w in 0..non_ctx_words.nrows(){
            if w != c_word {
                prob_sum += f64::exp(non_ctx_words.row(w).dot(&h.t()));
            }
        }
        let ctx_product = f64::exp(non_ctx_words.row(*ctx_word as usize).dot(&h.t()));
        ctx_product / prob_sum
    } 

    pub fn softmax(mut a: &Array<f64, Dim<[usize; 1]>>) -> Array<f64, Dim<[usize; 1]>>{
        let mut tot_sum = 0.;
        let mut ret_array = arr1(&vec![0.; a.len()]);
        for v in a {
            tot_sum += f64::exp(*v);
        }
        for (k, v) in a.iter().enumerate() {
            ret_array[k] = f64::exp(*v) / tot_sum;
        }
        ret_array
    }
}