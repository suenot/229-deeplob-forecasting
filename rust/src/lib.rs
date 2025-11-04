//! # DeepLOB Forecasting
//!
//! Implementation of the DeepLOB architecture for limit order book (LOB)
//! mid-price forecasting. Based on Zhang et al. (2019).
//!
//! Architecture: LOB Input -> Normalization -> Inception CNN -> LSTM -> 3-class Softmax

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

/// Number of LOB levels per side
pub const NUM_LEVELS: usize = 10;
/// Total features per snapshot: 10 levels x 2 sides x 2 (price + volume)
pub const NUM_FEATURES: usize = 40;
/// Number of output classes: up, stationary, down
pub const NUM_CLASSES: usize = 3;
/// Default smoothing horizon for label generation
pub const DEFAULT_HORIZON_K: usize = 5;

// ─── LOB Snapshot ───────────────────────────────────────────────────────────

/// A single limit order book snapshot with 10 levels per side.
#[derive(Debug, Clone)]
pub struct LOBSnapshot {
    pub ask_prices: Vec<f64>,
    pub ask_volumes: Vec<f64>,
    pub bid_prices: Vec<f64>,
    pub bid_volumes: Vec<f64>,
    pub timestamp: u64,
}

impl LOBSnapshot {
    /// Create a new LOB snapshot. Expects exactly 10 levels per side.
    pub fn new(
        ask_prices: Vec<f64>,
        ask_volumes: Vec<f64>,
        bid_prices: Vec<f64>,
        bid_volumes: Vec<f64>,
        timestamp: u64,
    ) -> anyhow::Result<Self> {
        if ask_prices.len() != NUM_LEVELS
            || ask_volumes.len() != NUM_LEVELS
            || bid_prices.len() != NUM_LEVELS
            || bid_volumes.len() != NUM_LEVELS
        {
            anyhow::bail!(
                "Each side must have exactly {} levels, got ask_p={}, ask_v={}, bid_p={}, bid_v={}",
                NUM_LEVELS,
                ask_prices.len(),
                ask_volumes.len(),
                bid_prices.len(),
                bid_volumes.len()
            );
        }
        Ok(Self {
            ask_prices,
            ask_volumes,
            bid_prices,
            bid_volumes,
            timestamp,
        })
    }

    /// Convert snapshot to a flat 40-element feature vector.
    /// Order: [ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ask_v2, bid_p2, bid_v2, ...]
    pub fn to_features(&self) -> Array1<f64> {
        let mut features = Array1::zeros(NUM_FEATURES);
        for i in 0..NUM_LEVELS {
            features[i * 4] = self.ask_prices[i];
            features[i * 4 + 1] = self.ask_volumes[i];
            features[i * 4 + 2] = self.bid_prices[i];
            features[i * 4 + 3] = self.bid_volumes[i];
        }
        features
    }

    /// Compute mid-price from best bid and ask.
    pub fn mid_price(&self) -> f64 {
        (self.ask_prices[0] + self.bid_prices[0]) / 2.0
    }
}

// ─── LOB Normalizer ─────────────────────────────────────────────────────────

/// Z-score normalizer that maintains running statistics for each of the 40 features.
#[derive(Debug, Clone)]
pub struct LOBNormalizer {
    pub mean: Array1<f64>,
    pub variance: Array1<f64>,
    pub count: usize,
    epsilon: f64,
}

impl LOBNormalizer {
    pub fn new() -> Self {
        Self {
            mean: Array1::zeros(NUM_FEATURES),
            variance: Array1::zeros(NUM_FEATURES),
            count: 0,
            epsilon: 1e-8,
        }
    }

    /// Update running statistics with a new feature vector using Welford's algorithm.
    pub fn update(&mut self, features: &Array1<f64>) {
        self.count += 1;
        let n = self.count as f64;
        let delta = features - &self.mean;
        self.mean = &self.mean + &(&delta / n);
        let delta2 = features - &self.mean;
        self.variance = &self.variance + &(&delta * &delta2);
    }

    /// Normalize a feature vector using current running statistics.
    pub fn normalize(&self, features: &Array1<f64>) -> Array1<f64> {
        if self.count < 2 {
            return features.clone();
        }
        let n = self.count as f64;
        let std_dev = (&self.variance / n).mapv(|v| (v.max(0.0)).sqrt() + self.epsilon);
        (features - &self.mean) / &std_dev
    }

    /// Fit the normalizer on a batch of feature vectors, then normalize them all.
    pub fn fit_transform(&mut self, data: &[Array1<f64>]) -> Vec<Array1<f64>> {
        for row in data {
            self.update(row);
        }
        data.iter().map(|row| self.normalize(row)).collect()
    }
}

impl Default for LOBNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Inception Module ───────────────────────────────────────────────────────

/// A 1D convolution kernel with bias.
#[derive(Debug, Clone)]
pub struct Conv1DKernel {
    /// Weights: shape (kernel_size, input_channels)
    pub weights: Array2<f64>,
    pub bias: f64,
    pub kernel_size: usize,
}

impl Conv1DKernel {
    /// Create a randomly initialized kernel (Xavier initialization).
    pub fn new_random(kernel_size: usize, input_channels: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (kernel_size + input_channels) as f64).sqrt();
        let weights =
            Array2::from_shape_fn((kernel_size, input_channels), |_| rng.gen::<f64>() * scale);
        Self {
            weights,
            bias: 0.0,
            kernel_size,
        }
    }

    /// Apply 1D convolution on a sequence with zero-padding to preserve length.
    /// Input: (seq_len, channels), Output: (seq_len,)
    pub fn forward(&self, input: &Array2<f64>) -> Array1<f64> {
        let seq_len = input.nrows();
        let pad = self.kernel_size / 2;
        let mut output = Array1::zeros(seq_len);

        for t in 0..seq_len {
            let mut sum = self.bias;
            for k in 0..self.kernel_size {
                let idx = t as isize + k as isize - pad as isize;
                if idx >= 0 && (idx as usize) < seq_len {
                    let row = input.row(idx as usize);
                    for c in 0..row.len().min(self.weights.ncols()) {
                        sum += row[c] * self.weights[[k, c]];
                    }
                }
            }
            output[t] = sum;
        }
        output
    }
}

/// Inception module: parallel convolutions at kernel sizes 1, 3, 5, concatenated.
#[derive(Debug, Clone)]
pub struct InceptionModule {
    pub conv1: Conv1DKernel,
    pub conv3: Conv1DKernel,
    pub conv5: Conv1DKernel,
    pub output_dim: usize,
}

impl InceptionModule {
    /// Create an inception module with given input channels.
    /// Output dimension = 3 (one feature per kernel scale).
    pub fn new(input_channels: usize) -> Self {
        Self {
            conv1: Conv1DKernel::new_random(1, input_channels),
            conv3: Conv1DKernel::new_random(3, input_channels),
            conv5: Conv1DKernel::new_random(5, input_channels),
            output_dim: 3,
        }
    }

    /// Forward pass: apply all three convolutions, ReLU, and concatenate.
    /// Input: (seq_len, input_channels), Output: (seq_len, 3)
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let seq_len = input.nrows();
        let out1 = self.conv1.forward(input).mapv(relu);
        let out3 = self.conv3.forward(input).mapv(relu);
        let out5 = self.conv5.forward(input).mapv(relu);

        let mut output = Array2::zeros((seq_len, 3));
        for t in 0..seq_len {
            output[[t, 0]] = out1[t];
            output[[t, 1]] = out3[t];
            output[[t, 2]] = out5[t];
        }
        output
    }
}

/// ReLU activation.
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

// ─── LSTM Layer ─────────────────────────────────────────────────────────────

/// A simple LSTM cell.
#[derive(Debug, Clone)]
pub struct LSTMLayer {
    pub input_size: usize,
    pub hidden_size: usize,
    // Gate weights: W_f, W_i, W_c, W_o each of shape (hidden_size, input_size + hidden_size)
    pub w_f: Array2<f64>,
    pub w_i: Array2<f64>,
    pub w_c: Array2<f64>,
    pub w_o: Array2<f64>,
    // Biases
    pub b_f: Array1<f64>,
    pub b_i: Array1<f64>,
    pub b_c: Array1<f64>,
    pub b_o: Array1<f64>,
}

impl LSTMLayer {
    /// Create a new LSTM layer with Xavier-initialized weights.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let combined = input_size + hidden_size;
        let scale = (2.0 / (combined + hidden_size) as f64).sqrt();

        let mut rand_matrix = |rows: usize, cols: usize| -> Array2<f64> {
            Array2::from_shape_fn((rows, cols), |_| rng.gen::<f64>() * scale - scale / 2.0)
        };

        Self {
            input_size,
            hidden_size,
            w_f: rand_matrix(hidden_size, combined),
            w_i: rand_matrix(hidden_size, combined),
            w_c: rand_matrix(hidden_size, combined),
            w_o: rand_matrix(hidden_size, combined),
            b_f: Array1::from_elem(hidden_size, 1.0), // Forget gate bias init to 1
            b_i: Array1::zeros(hidden_size),
            b_c: Array1::zeros(hidden_size),
            b_o: Array1::zeros(hidden_size),
        }
    }

    /// Process a sequence and return the final hidden state.
    /// Input: (seq_len, input_size), Output: (hidden_size,)
    pub fn forward(&self, input: &Array2<f64>) -> Array1<f64> {
        let seq_len = input.nrows();
        let mut h = Array1::zeros(self.hidden_size);
        let mut c = Array1::zeros(self.hidden_size);

        for t in 0..seq_len {
            let x_t = input.row(t);

            // Concatenate [h_{t-1}, x_t]
            let mut combined = Array1::zeros(self.hidden_size + self.input_size);
            for i in 0..self.hidden_size {
                combined[i] = h[i];
            }
            for i in 0..self.input_size {
                combined[self.hidden_size + i] = x_t[i];
            }

            // Gate computations
            let f_t = gate_forward(&self.w_f, &combined, &self.b_f, true);
            let i_t = gate_forward(&self.w_i, &combined, &self.b_i, true);
            let c_tilde = gate_forward(&self.w_c, &combined, &self.b_c, false);
            let o_t = gate_forward(&self.w_o, &combined, &self.b_o, true);

            // Cell state update
            c = &f_t * &c + &i_t * &c_tilde;

            // Hidden state update
            h = &o_t * &c.mapv(|x| x.tanh());
        }

        h
    }
}

/// Compute a gate output: sigmoid(W * x + b) or tanh(W * x + b).
fn gate_forward(
    w: &Array2<f64>,
    x: &Array1<f64>,
    b: &Array1<f64>,
    use_sigmoid: bool,
) -> Array1<f64> {
    let hidden_size = w.nrows();
    let mut output = Array1::zeros(hidden_size);
    for i in 0..hidden_size {
        let mut sum = b[i];
        for j in 0..x.len().min(w.ncols()) {
            sum += w[[i, j]] * x[j];
        }
        output[i] = if use_sigmoid { sigmoid(sum) } else { sum.tanh() };
    }
    output
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ─── DeepLOB Model ─────────────────────────────────────────────────────────

/// Prediction direction from DeepLOB.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PredictionClass {
    Up,
    Stationary,
    Down,
}

impl std::fmt::Display for PredictionClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PredictionClass::Up => write!(f, "UP"),
            PredictionClass::Stationary => write!(f, "STATIONARY"),
            PredictionClass::Down => write!(f, "DOWN"),
        }
    }
}

/// The full DeepLOB model: Normalizer -> 2x Inception -> LSTM -> FC -> Softmax.
#[derive(Debug, Clone)]
pub struct DeepLOBModel {
    pub normalizer: LOBNormalizer,
    pub inception1: InceptionModule,
    pub inception2: InceptionModule,
    pub lstm: LSTMLayer,
    /// Fully connected weights: (NUM_CLASSES, hidden_size)
    pub fc_weights: Array2<f64>,
    pub fc_bias: Array1<f64>,
    pub hidden_size: usize,
}

impl DeepLOBModel {
    /// Create a new DeepLOB model with given LSTM hidden size.
    pub fn new(hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (hidden_size + NUM_CLASSES) as f64).sqrt();
        let fc_weights = Array2::from_shape_fn((NUM_CLASSES, hidden_size), |_| {
            rng.gen::<f64>() * scale - scale / 2.0
        });

        Self {
            normalizer: LOBNormalizer::new(),
            inception1: InceptionModule::new(NUM_FEATURES),
            inception2: InceptionModule::new(3), // output of first inception
            lstm: LSTMLayer::new(3, hidden_size), // output of second inception -> LSTM
            fc_weights,
            fc_bias: Array1::zeros(NUM_CLASSES),
            hidden_size,
        }
    }

    /// Run inference on a sequence of LOB snapshots.
    /// Returns (class, probabilities) where probabilities are [p_up, p_stationary, p_down].
    pub fn predict(&mut self, snapshots: &[LOBSnapshot]) -> (PredictionClass, [f64; NUM_CLASSES]) {
        // Step 1: Convert to feature vectors and normalize
        let features: Vec<Array1<f64>> = snapshots.iter().map(|s| s.to_features()).collect();
        let normalized = self.normalizer.fit_transform(&features);

        // Step 2: Build input matrix (seq_len, 40)
        let seq_len = normalized.len();
        let mut input_matrix = Array2::zeros((seq_len, NUM_FEATURES));
        for (t, row) in normalized.iter().enumerate() {
            for j in 0..NUM_FEATURES {
                input_matrix[[t, j]] = row[j];
            }
        }

        // Step 3: Inception module 1
        let inc1_out = self.inception1.forward(&input_matrix);

        // Step 4: Inception module 2
        let inc2_out = self.inception2.forward(&inc1_out);

        // Step 5: LSTM
        let lstm_out = self.lstm.forward(&inc2_out);

        // Step 6: Fully connected + softmax
        let logits = self.fc_forward(&lstm_out);
        let probs = softmax(&logits);

        // Step 7: Determine class
        let class = if probs[0] >= probs[1] && probs[0] >= probs[2] {
            PredictionClass::Up
        } else if probs[2] >= probs[0] && probs[2] >= probs[1] {
            PredictionClass::Down
        } else {
            PredictionClass::Stationary
        };

        (class, [probs[0], probs[1], probs[2]])
    }

    fn fc_forward(&self, hidden: &Array1<f64>) -> Array1<f64> {
        let mut logits = Array1::zeros(NUM_CLASSES);
        for i in 0..NUM_CLASSES {
            let mut sum = self.fc_bias[i];
            for j in 0..self.hidden_size {
                sum += self.fc_weights[[i, j]] * hidden[j];
            }
            logits[i] = sum;
        }
        logits
    }
}

/// Softmax function with numerical stability.
pub fn softmax(logits: &Array1<f64>) -> Array1<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Array1<f64> = logits.mapv(|x| (x - max_val).exp());
    let sum: f64 = exp_vals.sum();
    exp_vals / sum
}

// ─── Smooth Label Generation ────────────────────────────────────────────────

/// Generate smooth labels for a series of mid-prices.
///
/// Returns a vector of labels: 1 (up), 0 (stationary), -1 (down).
/// Uses a forward-looking average over `horizon` steps and threshold `alpha`.
pub fn generate_smooth_labels(
    mid_prices: &[f64],
    horizon: usize,
    alpha: f64,
) -> Vec<i8> {
    let n = mid_prices.len();
    let mut labels = Vec::with_capacity(n);

    for t in 0..n {
        if t + horizon >= n {
            labels.push(0); // Not enough future data; label as stationary
            continue;
        }

        let current = mid_prices[t];
        if current.abs() < 1e-12 {
            labels.push(0);
            continue;
        }

        // Compute forward-looking average
        let future_avg: f64 = mid_prices[t + 1..=t + horizon].iter().sum::<f64>() / horizon as f64;
        let change = (future_avg - current) / current;

        if change > alpha {
            labels.push(1); // Up
        } else if change < -alpha {
            labels.push(-1); // Down
        } else {
            labels.push(0); // Stationary
        }
    }

    labels
}

/// Compute a threshold alpha that yields roughly balanced classes.
pub fn compute_balanced_alpha(mid_prices: &[f64], horizon: usize) -> f64 {
    let n = mid_prices.len();
    let mut changes: Vec<f64> = Vec::new();

    for t in 0..n.saturating_sub(horizon) {
        let current = mid_prices[t];
        if current.abs() < 1e-12 {
            continue;
        }
        let future_avg: f64 =
            mid_prices[t + 1..=t + horizon].iter().sum::<f64>() / horizon as f64;
        changes.push(((future_avg - current) / current).abs());
    }

    if changes.is_empty() {
        return 0.0001;
    }

    changes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Use the 33rd percentile as alpha so ~1/3 of samples are stationary
    let idx = changes.len() / 3;
    changes[idx]
}

// ─── Bybit Integration ─────────────────────────────────────────────────────

/// Bybit V5 orderbook API response structures.
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i64,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "a")]
    pub asks: Vec<[String; 2]>,
    #[serde(rename = "b")]
    pub bids: Vec<[String; 2]>,
    #[serde(rename = "ts")]
    pub timestamp: u64,
    #[serde(rename = "u")]
    pub update_id: u64,
}

/// Fetch the current orderbook from Bybit and return an LOBSnapshot.
pub fn fetch_bybit_orderbook(symbol: &str) -> anyhow::Result<LOBSnapshot> {
    let url = format!(
        "https://api.bybit.com/v5/market/orderbook?category=linear&symbol={}&limit=50",
        symbol
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let mut ask_prices = Vec::with_capacity(NUM_LEVELS);
    let mut ask_volumes = Vec::with_capacity(NUM_LEVELS);
    let mut bid_prices = Vec::with_capacity(NUM_LEVELS);
    let mut bid_volumes = Vec::with_capacity(NUM_LEVELS);

    for i in 0..NUM_LEVELS {
        if i < resp.result.asks.len() {
            ask_prices.push(resp.result.asks[i][0].parse::<f64>()?);
            ask_volumes.push(resp.result.asks[i][1].parse::<f64>()?);
        } else {
            ask_prices.push(0.0);
            ask_volumes.push(0.0);
        }

        if i < resp.result.bids.len() {
            bid_prices.push(resp.result.bids[i][0].parse::<f64>()?);
            bid_volumes.push(resp.result.bids[i][1].parse::<f64>()?);
        } else {
            bid_prices.push(0.0);
            bid_volumes.push(0.0);
        }
    }

    LOBSnapshot::new(ask_prices, ask_volumes, bid_prices, bid_volumes, resp.result.timestamp)
}

/// Asynchronous version of orderbook fetching.
pub async fn fetch_bybit_orderbook_async(symbol: &str) -> anyhow::Result<LOBSnapshot> {
    let url = format!(
        "https://api.bybit.com/v5/market/orderbook?category=linear&symbol={}&limit=50",
        symbol
    );

    let client = reqwest::Client::new();
    let resp: BybitResponse = client.get(&url).send().await?.json().await?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let mut ask_prices = Vec::with_capacity(NUM_LEVELS);
    let mut ask_volumes = Vec::with_capacity(NUM_LEVELS);
    let mut bid_prices = Vec::with_capacity(NUM_LEVELS);
    let mut bid_volumes = Vec::with_capacity(NUM_LEVELS);

    for i in 0..NUM_LEVELS {
        if i < resp.result.asks.len() {
            ask_prices.push(resp.result.asks[i][0].parse::<f64>()?);
            ask_volumes.push(resp.result.asks[i][1].parse::<f64>()?);
        } else {
            ask_prices.push(0.0);
            ask_volumes.push(0.0);
        }

        if i < resp.result.bids.len() {
            bid_prices.push(resp.result.bids[i][0].parse::<f64>()?);
            bid_volumes.push(resp.result.bids[i][1].parse::<f64>()?);
        } else {
            bid_prices.push(0.0);
            bid_volumes.push(0.0);
        }
    }

    LOBSnapshot::new(ask_prices, ask_volumes, bid_prices, bid_volumes, resp.result.timestamp)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(ask_base: f64, bid_base: f64, ts: u64) -> LOBSnapshot {
        let ask_prices: Vec<f64> = (0..NUM_LEVELS).map(|i| ask_base + i as f64 * 0.1).collect();
        let ask_volumes: Vec<f64> = (0..NUM_LEVELS).map(|i| 10.0 + i as f64).collect();
        let bid_prices: Vec<f64> = (0..NUM_LEVELS).map(|i| bid_base - i as f64 * 0.1).collect();
        let bid_volumes: Vec<f64> = (0..NUM_LEVELS).map(|i| 10.0 + i as f64).collect();
        LOBSnapshot::new(ask_prices, ask_volumes, bid_prices, bid_volumes, ts).unwrap()
    }

    #[test]
    fn test_lob_snapshot_features() {
        let snap = make_snapshot(100.05, 99.95, 1);
        let features = snap.to_features();
        assert_eq!(features.len(), NUM_FEATURES);
        // First feature should be ask_price[0]
        assert!((features[0] - 100.05).abs() < 1e-10);
        // Second feature should be ask_volume[0]
        assert!((features[1] - 10.0).abs() < 1e-10);
        // Third feature should be bid_price[0]
        assert!((features[2] - 99.95).abs() < 1e-10);
    }

    #[test]
    fn test_mid_price() {
        let snap = make_snapshot(100.10, 99.90, 1);
        let mid = snap.mid_price();
        assert!((mid - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalizer() {
        let snap1 = make_snapshot(100.0, 99.0, 1);
        let snap2 = make_snapshot(101.0, 100.0, 2);
        let snap3 = make_snapshot(102.0, 101.0, 3);

        let features: Vec<Array1<f64>> =
            vec![snap1.to_features(), snap2.to_features(), snap3.to_features()];

        let mut normalizer = LOBNormalizer::new();
        let normalized = normalizer.fit_transform(&features);

        assert_eq!(normalized.len(), 3);
        // After normalization, mean should be close to zero
        let mean_val: f64 = normalized.iter().map(|r| r[0]).sum::<f64>() / 3.0;
        assert!(mean_val.abs() < 1e-10, "Mean should be ~0, got {}", mean_val);
    }

    #[test]
    fn test_smooth_labels() {
        // Increasing prices -> should label as "up"
        let prices = vec![100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9];
        let labels = generate_smooth_labels(&prices, 5, 0.001);

        // First few should be "up" (1) since price is consistently rising
        assert_eq!(labels[0], 1);
        // Last elements don't have enough horizon
        assert_eq!(labels[prices.len() - 1], 0);
    }

    #[test]
    fn test_smooth_labels_down() {
        // Decreasing prices -> should label as "down"
        let prices = vec![100.0, 99.9, 99.8, 99.7, 99.6, 99.5, 99.4, 99.3, 99.2, 99.1];
        let labels = generate_smooth_labels(&prices, 5, 0.001);
        assert_eq!(labels[0], -1);
    }

    #[test]
    fn test_smooth_labels_stationary() {
        // Nearly flat prices -> should label as "stationary"
        let prices = vec![100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0];
        let labels = generate_smooth_labels(&prices, 5, 0.001);
        assert_eq!(labels[0], 0);
    }

    #[test]
    fn test_compute_balanced_alpha() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.01).collect();
        let alpha = compute_balanced_alpha(&prices, 5);
        assert!(alpha > 0.0, "Alpha should be positive");
        assert!(alpha < 0.01, "Alpha should be small for gradual trend");
    }

    #[test]
    fn test_inception_module() {
        let module = InceptionModule::new(NUM_FEATURES);
        let input = Array2::from_shape_fn((10, NUM_FEATURES), |(_, _)| 0.5);
        let output = module.forward(&input);
        assert_eq!(output.nrows(), 10);
        assert_eq!(output.ncols(), 3);
    }

    #[test]
    fn test_lstm_layer() {
        let lstm = LSTMLayer::new(3, 8);
        let input = Array2::from_shape_fn((10, 3), |(_, _)| 0.1);
        let output = lstm.forward(&input);
        assert_eq!(output.len(), 8);
        // LSTM output should be bounded by tanh (-1, 1)
        for &val in output.iter() {
            assert!(val.abs() <= 1.0 + 1e-10, "LSTM output should be in [-1,1], got {}", val);
        }
    }

    #[test]
    fn test_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);
        // Probabilities should sum to 1
        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-10, "Softmax should sum to 1, got {}", sum);
        // Higher logit should have higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_deeplob_model_predict() {
        let snapshots: Vec<LOBSnapshot> = (0..20)
            .map(|i| make_snapshot(100.0 + i as f64 * 0.01, 99.9 + i as f64 * 0.01, i as u64))
            .collect();

        let mut model = DeepLOBModel::new(8);
        let (class, probs) = model.predict(&snapshots);

        // Check probabilities sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Probs should sum to 1, got {}", sum);

        // Check all probabilities are non-negative
        for p in &probs {
            assert!(*p >= 0.0, "Probability should be >= 0, got {}", p);
        }

        // Check class is valid
        assert!(
            class == PredictionClass::Up
                || class == PredictionClass::Stationary
                || class == PredictionClass::Down
        );
    }

    #[test]
    fn test_snapshot_invalid_levels() {
        let result = LOBSnapshot::new(
            vec![1.0, 2.0], // only 2 levels
            vec![1.0; 10],
            vec![1.0; 10],
            vec![1.0; 10],
            0,
        );
        assert!(result.is_err());
    }
}
