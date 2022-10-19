#[cfg(test)]
mod train_test {
    use word_embeddings::{CustomActivationFunction, CustomProbFunctionType, SkipGram};
    use ndarray::{ArrayBase, OwnedRepr, Dim};
    use serde::{Deserialize, Deserializer, Serialize, de::{self, SeqAccess}};
    use std::marker::PhantomData;
    use std::fmt;
    
    type T = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;

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


    
    const HU_MULTIPLIER: f64 =  1.379289341667417;
    static mut MODEL: SkipGram = SkipGram {
        w_size: 3,
        d: 80,
        lr: 0.005146257850989499,
        prob_function: CustomProbFunctionType::Softmax,
        activation_fn: CustomActivationFunction::Sigmoid,
        batches: 32,
        train_split: 0.85,
        epochs: 20,
        data: None,
        k: None,
        beta: 0.9,
    };
    
    #[test]
    fn train() {
        let mut trained_weights: Vec<T> = Default::default();
        let mut avg_error: f64 = 0.;
        unsafe {
            let ctx_map = MODEL.preprocess_data("/Users/cinderella/Documents/word-embeddings/tests/word_set.txt", false).unwrap(); // in case its needed
            let data = MODEL.data.clone();
            MODEL.d = (HU_MULTIPLIER * data.unwrap().len() as f64) as i32;
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
    }

/* 
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
    
    } */
}

