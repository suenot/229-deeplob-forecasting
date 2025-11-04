# Chapter 271: DeepLOB Forecasting

## 1. Introduction

The Limit Order Book (LOB) is the fundamental data structure that drives price formation in modern electronic markets. It records all outstanding buy and sell orders at various price levels, providing a real-time snapshot of supply and demand. Predicting the short-term direction of the mid-price---the average of the best bid and best ask prices---has been a longstanding challenge in quantitative finance and market microstructure research.

**DeepLOB** (Zhang et al., 2019) is a landmark deep learning architecture specifically designed for LOB-based mid-price forecasting. The model combines Convolutional Neural Networks (CNNs) with Long Short-Term Memory (LSTM) networks to capture both spatial patterns across price levels and temporal dependencies across time. Its key innovation is the use of **inception modules** (inspired by GoogLeNet) that apply parallel convolutions at multiple scales, enabling the model to detect features ranging from narrow spread dynamics to broader supply/demand imbalances.

DeepLOB has become one of the most cited and reproduced models in the LOB forecasting literature, consistently outperforming traditional handcrafted features and simpler neural architectures on benchmark datasets such as FI-2010. In this chapter, we explore its architecture, mathematical foundations, and provide a Rust implementation that integrates with live Bybit orderbook data for cryptocurrency trading.

### Why DeepLOB Matters for Trading

1. **Data-driven feature extraction**: Instead of manually engineering LOB features (order imbalance, spread, depth ratios), DeepLOB learns optimal representations directly from raw LOB data.
2. **Multi-scale pattern recognition**: The inception modules capture both local (single price level) and global (cross-level) patterns simultaneously.
3. **Temporal modeling**: The LSTM component captures how LOB states evolve over time, identifying momentum and mean-reversion regimes.
4. **Production applicability**: The architecture is lightweight enough for low-latency inference, making it suitable for real-time trading systems.

## 2. Mathematical Foundations

### 2.1 LOB Input Representation

The LOB at time $t$ is represented as a matrix $X_t \in \mathbb{R}^{T \times 40}$, where $T$ is the lookback window (number of historical snapshots) and 40 is the feature dimension:

$$X_t = \begin{bmatrix} p_1^{ask}(t) & v_1^{ask}(t) & p_1^{bid}(t) & v_1^{bid}(t) & \cdots & p_{10}^{ask}(t) & v_{10}^{ask}(t) & p_{10}^{bid}(t) & v_{10}^{bid}(t) \end{bmatrix}$$

For each of the 10 price levels on each side of the book, we record:
- $p_i^{ask}(t)$: The $i$-th best ask price at time $t$
- $v_i^{ask}(t)$: The volume at the $i$-th best ask level
- $p_i^{bid}(t)$: The $i$-th best bid price at time $t$
- $v_i^{bid}(t)$: The volume at the $i$-th best bid level

This gives us $10 \times 2 \times 2 = 40$ features per timestamp.

### 2.2 Normalization

Before feeding data into the network, we apply z-score normalization across the time dimension for each feature $j$:

$$\hat{x}_{t,j} = \frac{x_{t,j} - \mu_j}{\sigma_j + \epsilon}$$

where $\mu_j$ and $\sigma_j$ are the mean and standard deviation of feature $j$ computed over a rolling window, and $\epsilon = 10^{-8}$ is a small constant for numerical stability.

### 2.3 Inception Module

The core building block of DeepLOB's CNN component is the inception module, which applies multiple convolution operations in parallel:

$$h_t^{(k)} = \text{ReLU}\left(\sum_{s \in \{1, 3, 5\}} W_s^{(k)} * x_t^{(k-1)} + b_s^{(k)}\right)$$

where:
- $W_s^{(k)}$ is the convolution kernel of size $s$ at layer $k$
- $*$ denotes the convolution operation
- The outputs from different kernel sizes are concatenated along the channel dimension

This multi-scale approach allows the model to detect:
- **Scale 1 (1x1 conv)**: Point-wise transformations of individual features
- **Scale 3 (1x3 conv)**: Local patterns across adjacent price levels
- **Scale 5 (1x5 conv)**: Broader patterns spanning multiple price levels

### 2.4 LSTM Component

After the CNN layers extract spatial features, the LSTM processes the sequence of extracted features to capture temporal dynamics:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

where:
- $f_t$, $i_t$, $o_t$ are the forget, input, and output gates
- $C_t$ is the cell state
- $\sigma$ is the sigmoid activation function
- $\odot$ denotes element-wise multiplication

### 2.5 Smooth Labeling (Three-Class Classification)

DeepLOB predicts one of three classes: **up**, **down**, or **stationary**. The labels are generated using a smoothed mid-price change over a future horizon $k$:

$$m_t = \frac{p_t^{ask,1} + p_t^{bid,1}}{2}$$

$$\bar{m}_{t+k} = \frac{1}{k} \sum_{i=1}^{k} m_{t+i}$$

$$l_t = \frac{\bar{m}_{t+k} - m_t}{m_t}$$

The label is then assigned based on a threshold $\alpha$ (typically set so that each class has roughly equal representation):

$$y_t = \begin{cases} 1 \text{ (up)} & \text{if } l_t > \alpha \\ -1 \text{ (down)} & \text{if } l_t < -\alpha \\ 0 \text{ (stationary)} & \text{otherwise} \end{cases}$$

The smoothing over $k$ future steps (typically $k=5$) reduces noise and makes the prediction target more robust.

### 2.6 Loss Function

The model is trained with categorical cross-entropy loss with optional class weights to handle imbalanced datasets:

$$\mathcal{L} = -\sum_{c \in \{-1, 0, 1\}} w_c \cdot y_c \log(\hat{y}_c)$$

where $w_c$ is the weight for class $c$ and $\hat{y}_c$ is the softmax output for class $c$.

## 3. Three-Class Prediction with Smooth Labeling

### 3.1 Why Three Classes?

Binary up/down classification ignores the reality that prices often remain virtually unchanged in the short term. The "stationary" class captures these periods, which has several benefits:

1. **Reduced false signals**: The model avoids generating spurious trading signals during calm markets.
2. **Better calibrated confidence**: The model can express uncertainty through the stationary class instead of forcing a directional prediction.
3. **Transaction cost awareness**: Stationary predictions naturally filter out moves too small to profit from after transaction costs.

### 3.2 Threshold Selection

The threshold $\alpha$ can be set using several strategies:

- **Fixed percentile**: Choose $\alpha$ so that each class contains approximately one-third of the training samples.
- **Transaction cost-based**: Set $\alpha$ equal to the expected round-trip transaction cost (spread + fees), ensuring only profitable moves are labeled as directional.
- **Volatility-adjusted**: Scale $\alpha$ by recent volatility, adapting to changing market conditions.

### 3.3 Label Smoothing in Practice

The forward-looking average $\bar{m}_{t+k}$ with $k=5$ serves several purposes:

- **Noise filtering**: Individual tick-level mid-price changes are extremely noisy; averaging smooths out microstructure noise.
- **Horizon flexibility**: By varying $k$, traders can target different prediction horizons (e.g., $k=1$ for ultra-short, $k=50$ for short-term).
- **Label stability**: Smoothed labels change less frequently, making the classification task more learnable.

## 4. Rust Implementation

Our Rust implementation provides the following components:

### 4.1 LOB Data Structures

```rust
pub struct LOBSnapshot {
    pub ask_prices: Vec<f64>,   // 10 levels
    pub ask_volumes: Vec<f64>,  // 10 levels
    pub bid_prices: Vec<f64>,   // 10 levels
    pub bid_volumes: Vec<f64>,  // 10 levels
    pub timestamp: u64,
}
```

The `LOBSnapshot` struct captures the full 10-level orderbook at a given moment. The `LOBNormalizer` handles z-score normalization, maintaining running statistics across the feature dimension.

### 4.2 Inception Module

The inception module applies three parallel 1D convolutions with kernel sizes 1, 3, and 5, then concatenates the results. Each convolution branch includes zero-padding to preserve the temporal dimension and ReLU activation for non-linearity.

### 4.3 LSTM Layer

The LSTM layer processes the sequence of CNN-extracted features. It maintains hidden state and cell state across time steps, implementing the standard LSTM equations with forget, input, and output gates.

### 4.4 DeepLOB Classifier

The full model chains:
1. Input normalization (40 features)
2. Two inception modules (multi-scale feature extraction)
3. LSTM layer (temporal modeling)
4. Fully connected layer with softmax (3-class output)

### 4.5 Key Implementation Details

- **Weight initialization**: Xavier/Glorot initialization for stable training.
- **Numerical stability**: All exponential and logarithmic operations include epsilon guards.
- **Memory efficiency**: The model operates on fixed-size arrays where possible, avoiding unnecessary allocations.

## 5. Bybit Data Integration

### 5.1 Orderbook API

Bybit provides REST and WebSocket endpoints for orderbook data. Our implementation uses the REST API for simplicity:

```
GET https://api.bybit.com/v5/market/orderbook?category=linear&symbol=BTCUSDT&limit=50
```

This returns up to 50 levels of the orderbook, from which we extract the top 10 bid and ask levels.

### 5.2 Data Pipeline

The integration pipeline consists of:

1. **Fetching**: Periodic REST calls to retrieve the current orderbook state.
2. **Parsing**: Extracting price and volume arrays from the JSON response.
3. **Normalization**: Applying z-score normalization using rolling statistics.
4. **Windowing**: Maintaining a sliding window of $T$ recent snapshots.
5. **Inference**: Running the DeepLOB model on the current window.
6. **Signal generation**: Converting the three-class output to a trading signal.

### 5.3 Live Trading Considerations

When deploying DeepLOB in production:

- **Latency**: The model should complete inference within the orderbook update interval (typically 100ms for Bybit).
- **Data quality**: Handle missing levels, stale data, and exchange disconnections gracefully.
- **Regime detection**: Monitor model confidence and reduce position sizing when the stationary class dominates predictions.
- **Calibration**: Periodically retrain or fine-tune the model as market microstructure evolves.

### 5.4 Cryptocurrency-Specific Considerations

Crypto LOB data differs from traditional equities in several ways:

- **24/7 markets**: No opening/closing auctions, continuous trading.
- **Higher volatility**: Wider spreads and more aggressive price movements.
- **Thinner books**: Fewer resting orders, especially in altcoins.
- **Fragmentation**: Liquidity split across multiple exchanges.

These factors mean that $\alpha$ thresholds and normalization windows may need different calibration compared to equity markets.

## 6. Key Takeaways

1. **DeepLOB combines CNNs and LSTMs** to learn spatial and temporal patterns in limit order book data, achieving state-of-the-art mid-price forecasting without manual feature engineering.

2. **Inception modules are critical**: The multi-scale convolution approach captures patterns at different granularities---from single price level dynamics to cross-level supply/demand imbalances.

3. **Three-class prediction with smooth labeling** provides more actionable signals than binary classification, naturally filtering out noise and unprofitable trades.

4. **The 40-feature LOB representation** (10 levels x 2 sides x 2 attributes) strikes a balance between information richness and computational tractability.

5. **Normalization matters**: Z-score normalization across features is essential for stable training and cross-asset generalization.

6. **Horizon parameter $k$ controls the trade-off** between prediction accuracy (higher $k$) and signal timeliness (lower $k$). $k=5$ is a common default.

7. **Rust implementation enables low-latency inference**, making DeepLOB practical for real-time trading applications where microseconds matter.

8. **Live deployment requires careful engineering** beyond the model itself: data pipeline reliability, regime detection, and continuous calibration are all essential for production systems.

## References

- Zhang, Z., Zohren, S., & Roberts, S. (2019). "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books." *IEEE Transactions on Signal Processing*, 67(11), 3001-3012.
- Szegedy, C., et al. (2015). "Going Deeper with Convolutions." *CVPR*.
- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780.
- Sirignano, J., & Cont, R. (2019). "Universal Features of Price Formation in Financial Markets." *PLOS ONE*.
- Ntakaris, A., et al. (2018). "Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods." *Journal of Forecasting*.
