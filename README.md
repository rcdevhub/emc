# Ethical Model Calibration
This repo contains the code for the project *Ethical Model Calibration - Diagnosing and Mitigating Hidden Bias in Healthcare Models*.

The accompanying MSc project dissertation is available <a href="https://www.dropbox.com/s/4o7s824vz7y5st1/COMP0158_LPZY4.pdf?dl=0" target="_blank">here</a>. A shorter draft summary paper is <a href="https://www.dropbox.com/s/lh5n0cc37oacktp/Ethical_Model_Calibration_article-draft-watermark.pdf?dl=0" target="_blank">here</a>.

## Objective
The objective of this project was to provide a framework to identify and mitigate bias in healthcare machine learning models, based on model performance in the latent space.
## Data
This project used two custom datasets from the UK Biobank.
## How to use
Add filepaths to `Main.py` and `Parameters.py`<br/>
Run the main experiment loop contained in `Main.py`<br/>
Run additional experiments contained in `Experiments_[NAME].py`<br/>
Parameters are contained in `Parameters.py`<br/>
Functions are contained in `Functions.py` and for additional experiments in `Functions_BVAE.py`
