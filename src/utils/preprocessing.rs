use std::{collections::HashMap, cmp::min};


pub fn build_context(c: &Vec<&str>, w_size: &i32, content_array: &Vec<String>) -> Result<HashMap<i32, Vec<i32>>, String> {
    let mut context_map = HashMap::<i32, Vec<i32>>::new(); // Key: index in cleaned dataset, Values: context indices
    
    println!("Building context map, this could take some time...");
    for w_ind in 0..c.len() {
        let checked_sub = match w_ind.checked_sub(*w_size as usize) {
            Some(s) => s,
            None => 0,
        };
        let ctx_range = checked_sub..min(c.len(), w_ind+(*w_size as usize));
        let mid_word_ind = content_array.iter().position(|w| *w==c[w_ind]).unwrap() as i32;
        for ctx_ind in ctx_range {
            if ctx_ind == w_ind {
                continue;
            }
            let ctx_word = content_array.iter().position(|w| *w==c[ctx_ind]).unwrap() as i32;
            let ctx_entry = context_map.entry(ctx_word).or_insert(Vec::new());
            if !ctx_entry.contains(&mid_word_ind) {
                ctx_entry.push(mid_word_ind);
            }
        }
    }
    println!("Context map built successfully");
    Ok(context_map)
}
