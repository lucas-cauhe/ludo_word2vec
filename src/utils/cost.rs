use super::types::*;

pub fn log_probability(ckey: &i32, w_indow: &[i32], w_output: &ArrT, hidden: &ArrT1, curr_len: usize) -> f64 {
    let mut brute_cost = 0.;
    
    for ctxkey in w_indow {
        let prob = prob_function(*ckey as usize, ctxkey, w_output, curr_len, hidden);
        brute_cost += f64::ln(prob);
    }
    brute_cost / curr_len as f64

}

fn prob_function(c_word: usize, ctx_word: &i32, w_output: &ArrT, d_len: usize, hidden: &ArrT1) -> f64 {
    let mut prob_sum = 0.;
    for w in 0..d_len{
        if w != c_word {
            prob_sum += f64::exp(w_output.column(w).dot(hidden));
        }
    }
    let ctx_product = f64::exp(w_output.column(*ctx_word as usize).dot(hidden));
    ctx_product / prob_sum
} 