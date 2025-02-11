## Installation (crc)

First make sure that you have anaconda installed 
```
cd ~
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
```
This may take a bit. After the anaconda is installed on the crc we can create an conda environment with 

```
conda create -n pmb python=3.9
```
Then we can install the required packages with 
```
cd /ix1/tibrahim/shared/tibrahim_drj21/03-PMB/alignment_code 
conda activate pmb
pip3 install -r requirements.txt
```
