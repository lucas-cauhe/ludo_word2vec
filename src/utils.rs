pub mod cost;
pub mod enums;
pub mod funcs;
pub mod initialization;
pub mod preprocessing;
pub mod types;

/* MISCELLANEOUS UTILS */

use plotters::prelude::*;

pub fn plot(train_errs: &[f64], test_errs: &[f64]) {
    
    let root_area = BitMapBackend::new("/Users/cinderella/Documents/word-embeddings/data/plot.png", (600, 400)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();
    
    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 100.0)
        .set_label_area_size(LabelAreaPosition::Bottom, 40.0)
        .set_label_area_size(LabelAreaPosition::Right, 100.0)
        .set_label_area_size(LabelAreaPosition::Top, 40.0)
        .caption("Training Plot", ("sans-serif", 40.0))
        .build_cartesian_2d(0..20, 0.0..4000.0)
        .unwrap();
    ctx.configure_mesh().draw().unwrap();

    ctx.draw_series(
        LineSeries::new((0..).zip(train_errs.iter()).map(|(idx, y)| {(idx, -y)}),&BLUE)
    ).unwrap();
    ctx.draw_series(
        LineSeries::new((0..).zip(test_errs.iter()).map(|(idx, y)| {(idx, -y)}),&RED)
    ).unwrap();
}

use self::types::ArrT;

pub fn show_weights(network_weights: &[ArrT]){
    let input_weights = &network_weights[0];
    let i_vec = vec![input_weights.row(0), 
    input_weights.row(10), 
    input_weights.row(50), 
    input_weights.row(100)];
    println!("Current input weights sample: {:?}", i_vec);
    let output_weights = &network_weights[1];
    let o_vec = vec![output_weights.row(0), 
    output_weights.row(10), 
    output_weights.row(50), 
    output_weights.row(100)];
    println!("Current input weights sample: {:?}", o_vec);
}