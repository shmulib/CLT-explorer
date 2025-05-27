# Interactive CLT Visualizer

An interactive Streamlit app that compares the distribution of scaled variables (`nX`) with the sum or mean of `n` i.i.d. variables (`ΣXᵢ` or `mean(Xᵢ)`), showcasing how the Central Limit Theorem plays out across different distributions.

## Features

- Choose from Uniform(0,1), Normal(0,1), or Exponential(λ=1)
- Toggle between plotting 5X vs ΣXᵢ or X vs mean(Xᵢ)
- Set the number of samples and animation speed
- Dynamically adjust the confidence level for bounding the visual range

## Run It Locally

```bash
pip install -r requirements.txt
streamlit run app.py
