//! # DeepLOB Trading Example
//!
//! Demonstrates fetching BTCUSDT orderbook snapshots from Bybit,
//! building a LOB feature matrix, and running DeepLOB inference.

use deeplob_forecasting::{
    compute_balanced_alpha, fetch_bybit_orderbook, generate_smooth_labels, DeepLOBModel,
    LOBSnapshot, NUM_LEVELS,
};
use std::thread;
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    println!("=== DeepLOB Forecasting - Trading Example ===\n");
    println!("Fetching BTCUSDT orderbook snapshots from Bybit...\n");

    let num_snapshots = 20;
    let mut snapshots: Vec<LOBSnapshot> = Vec::with_capacity(num_snapshots);
    let mut mid_prices: Vec<f64> = Vec::with_capacity(num_snapshots);

    // Collect orderbook snapshots
    for i in 0..num_snapshots {
        match fetch_bybit_orderbook("BTCUSDT") {
            Ok(snap) => {
                let mid = snap.mid_price();
                println!(
                    "Snapshot {}/{}: mid_price={:.2}, best_ask={:.2}, best_bid={:.2}, spread={:.2}",
                    i + 1,
                    num_snapshots,
                    mid,
                    snap.ask_prices[0],
                    snap.bid_prices[0],
                    snap.ask_prices[0] - snap.bid_prices[0],
                );
                mid_prices.push(mid);
                snapshots.push(snap);
            }
            Err(e) => {
                println!("Snapshot {}/{}: Failed to fetch: {}", i + 1, num_snapshots, e);
                println!("Using synthetic data instead...");
                return run_with_synthetic_data();
            }
        }

        if i < num_snapshots - 1 {
            thread::sleep(Duration::from_millis(500));
        }
    }

    // Generate smooth labels for analysis
    println!("\n--- Smooth Label Analysis ---");
    let alpha = compute_balanced_alpha(&mid_prices, 5);
    println!("Computed balanced alpha: {:.8}", alpha);
    let labels = generate_smooth_labels(&mid_prices, 5, alpha);
    let up_count = labels.iter().filter(|&&l| l == 1).count();
    let down_count = labels.iter().filter(|&&l| l == -1).count();
    let stationary_count = labels.iter().filter(|&&l| l == 0).count();
    println!(
        "Label distribution: UP={}, STATIONARY={}, DOWN={}",
        up_count, stationary_count, down_count
    );

    // Run DeepLOB inference
    println!("\n--- DeepLOB Inference ---");
    let mut model = DeepLOBModel::new(16);
    let (prediction, probs) = model.predict(&snapshots);

    println!("Prediction: {}", prediction);
    println!(
        "Probabilities: UP={:.4}, STATIONARY={:.4}, DOWN={:.4}",
        probs[0], probs[1], probs[2]
    );

    // Trading signal interpretation
    println!("\n--- Trading Signal ---");
    match prediction {
        deeplob_forecasting::PredictionClass::Up => {
            println!("Signal: LONG - Model predicts mid-price increase");
        }
        deeplob_forecasting::PredictionClass::Down => {
            println!("Signal: SHORT - Model predicts mid-price decrease");
        }
        deeplob_forecasting::PredictionClass::Stationary => {
            println!("Signal: HOLD - Model predicts no significant price movement");
        }
    }

    // LOB depth analysis
    println!("\n--- LOB Depth Analysis ---");
    if let Some(last) = snapshots.last() {
        let total_ask_vol: f64 = last.ask_volumes.iter().sum();
        let total_bid_vol: f64 = last.bid_volumes.iter().sum();
        let imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol);
        println!("Total ask volume (10 levels): {:.4}", total_ask_vol);
        println!("Total bid volume (10 levels): {:.4}", total_bid_vol);
        println!("Volume imbalance: {:.4} (positive = more bids)", imbalance);
    }

    println!("\n=== Example Complete ===");
    Ok(())
}

/// Fallback: run with synthetic data when the exchange API is unavailable.
fn run_with_synthetic_data() -> anyhow::Result<()> {
    println!("\n--- Running with Synthetic LOB Data ---\n");

    let num_snapshots = 20;
    let mut snapshots: Vec<LOBSnapshot> = Vec::with_capacity(num_snapshots);
    let mut mid_prices: Vec<f64> = Vec::with_capacity(num_snapshots);

    // Generate synthetic LOB data with a slight uptrend
    let base_price = 50000.0;
    for i in 0..num_snapshots {
        let drift = i as f64 * 0.5;
        let ask_base = base_price + drift + 0.5;
        let bid_base = base_price + drift - 0.5;

        let ask_prices: Vec<f64> = (0..NUM_LEVELS)
            .map(|l| ask_base + l as f64 * 0.1)
            .collect();
        let ask_volumes: Vec<f64> = (0..NUM_LEVELS).map(|l| 1.0 + l as f64 * 0.5).collect();
        let bid_prices: Vec<f64> = (0..NUM_LEVELS)
            .map(|l| bid_base - l as f64 * 0.1)
            .collect();
        let bid_volumes: Vec<f64> = (0..NUM_LEVELS).map(|l| 1.0 + l as f64 * 0.5).collect();

        let snap = LOBSnapshot::new(ask_prices, ask_volumes, bid_prices, bid_volumes, i as u64)?;
        let mid = snap.mid_price();
        println!(
            "Snapshot {}/{}: mid_price={:.2}, spread={:.2}",
            i + 1,
            num_snapshots,
            mid,
            snap.ask_prices[0] - snap.bid_prices[0],
        );
        mid_prices.push(mid);
        snapshots.push(snap);
    }

    // Label generation
    println!("\n--- Smooth Label Analysis ---");
    let alpha = compute_balanced_alpha(&mid_prices, 5);
    println!("Computed balanced alpha: {:.8}", alpha);
    let labels = generate_smooth_labels(&mid_prices, 5, alpha);
    let up_count = labels.iter().filter(|&&l| l == 1).count();
    let down_count = labels.iter().filter(|&&l| l == -1).count();
    let stationary_count = labels.iter().filter(|&&l| l == 0).count();
    println!(
        "Label distribution: UP={}, STATIONARY={}, DOWN={}",
        up_count, stationary_count, down_count
    );

    // DeepLOB inference
    println!("\n--- DeepLOB Inference ---");
    let mut model = DeepLOBModel::new(16);
    let (prediction, probs) = model.predict(&snapshots);

    println!("Prediction: {}", prediction);
    println!(
        "Probabilities: UP={:.4}, STATIONARY={:.4}, DOWN={:.4}",
        probs[0], probs[1], probs[2]
    );

    println!("\n--- Trading Signal ---");
    match prediction {
        deeplob_forecasting::PredictionClass::Up => {
            println!("Signal: LONG - Model predicts mid-price increase");
        }
        deeplob_forecasting::PredictionClass::Down => {
            println!("Signal: SHORT - Model predicts mid-price decrease");
        }
        deeplob_forecasting::PredictionClass::Stationary => {
            println!("Signal: HOLD - Model predicts no significant price movement");
        }
    }

    println!("\n=== Synthetic Example Complete ===");
    Ok(())
}
