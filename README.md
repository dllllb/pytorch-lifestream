# Setup

```sh
# download Miniconda from https://conda.io/
curl -OL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# execute intall file
bash Miniconda*.sh
# relogin to apply PATH changes
exit

# install pytorch
conda install pytorch -c pytorch
# check CUDA version
python -c 'import torch; print(torch.version.cuda)'
# chech torch GPU
# See question: https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
python -c 'import torch; print(torch.rand(2,3).cuda())'

# clone repo
git clone git@bitbucket.org:dllllb/dltranz.git

# install dependencies
cd dltranz
pip install -r requirements.txt

```

# Run scenario

```sh
cd experiments

# See README.md from the specific project
```

# Final results
```

LightGBM:                                     mean t_int_l t_int_h    std                           values
    Designed features:                                                                                    
        Age group   |   0.626 \pm 0.004   | 0.6255  0.6215  0.6296 0.0029  [0.622 0.624 0.624 0.627 0.630]
        Gender      |   0.875 \pm 0.004   | 0.8798  0.8751  0.8844 0.0033  [0.874 0.879 0.881 0.882 0.882]
        Assessment  |   0.591 \pm 0.003   | 0.5909  0.5872  0.5946 0.0027  [0.588 0.588 0.591 0.593 0.594]
        Retail      |   0.545 \pm 0.001   | 0.5454  0.5439  0.5469 0.0011  [0.544 0.545 0.545 0.546 0.547]
    CPC embeddings:         
        Age group   |   0.595 \pm 0.004   | 0.6025  0.6004  0.6047 0.0015  [0.601 0.601 0.603 0.604 0.604]
        Gender      |   0.848 \pm 0.004   | 0.8572  0.8545  0.8600 0.0020  [0.855 0.856 0.857 0.858 0.860]
        Assessment  |   0.584 \pm 0.004   | 0.5838  0.5790  0.5886 0.0035  [0.578 0.584 0.585 0.586 0.587]
        Retail      |   0.527 \pm 0.001   | 0.5265  0.5252  0.5278 0.0009  [0.525 0.526 0.527 0.527 0.528]
    MeLES embeddings:       
        Age group   |   0.639 \pm 0.006   | 0.6349  0.6302  0.6395 0.0034  [0.632 0.632 0.634 0.637 0.640]
        Gender      |   0.872 \pm 0.005   | 0.8821  0.8801  0.8840 0.0014  [0.880 0.882 0.882 0.882 0.884]
        Assessment  |   0.604 \pm 0.003   | 0.6041  0.5994  0.6088 0.0034  [0.598 0.604 0.605 0.606 0.607]
        Retail      |   0.544 \pm 0.001   | 0.5439  0.5421  0.5457 0.0013  [0.542 0.544 0.544 0.545 0.545]
                              
Scores:                       
    Supervised learning:    
        Age group   |   0.631 \pm 0.010   | 0.6331  0.6242  0.6419 0.0064  [0.625 0.630 0.634 0.634 0.643]
        Gender      |   0.871 \pm 0.007   | 0.8741  0.8654  0.8828 0.0063  [0.865 0.872 0.873 0.879 0.881]
        Assessment  |   0.601 \pm 0.006   | 0.6010  0.5948  0.6072 0.0045  [0.596 0.600 0.600 0.601 0.608]
        Retail      |   0.543 \pm 0.002   | 0.5425  0.5403  0.5447 0.0016  [0.540 0.542 0.542 0.544 0.544]
    CPC fine-tuning:        
        Age group   |   0.621 \pm 0.007   | 0.6171  0.6144  0.6197 0.0019  [0.614 0.617 0.617 0.618 0.619]
        Gender      |   0.873 \pm 0.007   | 0.8777  0.8726  0.8829 0.0037  [0.874 0.875 0.878 0.879 0.883]
        Assessment  |   0.611 \pm 0.005   | 0.6115  0.6047  0.6184 0.0049  [0.603 0.611 0.613 0.615 0.616]
        Retail      |   0.546 \pm 0.003   | 0.5461  0.5429  0.5492 0.0023  [0.542 0.546 0.546 0.547 0.548]
    MeLES fine-tuning:      
        Age group   |   0.643 \pm 0.007   | 0.6406  0.6329  0.6483 0.0055  [0.633 0.640 0.641 0.641 0.649]
        Gender      |   0.888 \pm 0.002   | 0.8959  0.8910  0.9009 0.0036  [0.890 0.896 0.898 0.898 0.898]
        Assessment  |   0.611 \pm 0.005   | 0.6113  0.6008  0.6218 0.0076  [0.600 0.611 0.612 0.613 0.621]
        Retail      |   0.549 \pm 0.001   | 0.5490  0.5479  0.5500 0.0008  [0.548 0.549 0.549 0.550 0.550]

```
    