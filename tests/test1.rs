use word_embeddings::{CustomActivationFunction, CustomProbFunctionType, SkipGram};

#[test]
fn trains_ok() {
    let mut model = SkipGram {
        w_size: 2,
        d: 5,
        lr: 0.1,
        prob_function: CustomProbFunctionType::Softmax,
        activation_fn: CustomActivationFunction::Sigmoid,
        dropout: false,
        batches: 15,
        data: None,
        k: None
    };

    let ctxMap = model.preprocess_data("/Users/cinderella/Documents/word-embeddings/tests/word_set.txt", false).unwrap(); // in case its needed
    println!("{:?}", model.data);
    let trained_weights = model.train(&ctxMap).expect("Smth went wrong");

    println!("Trained input weights -> {:}", trained_weights.0);
    println!("Trained output weights  -> {:}", trained_weights.1);

}