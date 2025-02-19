
## Logging into crc
You can log into CRC with terminal with

ssh ```pitt_user_name@h2p.crc.pitt.edu```

Or you can log into CRC with the remote ssh plug-in using vscode instructions can be found [here](https://crc-pages.pitt.edu/user-manual/slurm/vscode/)

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

## Usage (crc)
Make sure that you have ```angle_pos``` file first. For example:
```
2024.12.16-17.11.39/
├── ADRC_137/
├── CW24-36/
└── angle_pos
```
If you want to align case ADRC_137 you can run the following command under the alignment_code folder with

```python alignment_annotated.py 2024.12.16-17.11.39 ADRC_137 CW24-36```

After this command is done running, you will see a pptx file under the folder. For example:
```
2024.12.16-17.11.39/
├── ADRC_137/
├── CW24-36/
├── angle_pos
└── CW24-36.pptx
```
