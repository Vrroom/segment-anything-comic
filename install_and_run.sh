echo "Please enter the full path to your Miniconda3 installation:"
read miniconda_path
source "$miniconda_path/etc/profile.d/conda.sh"

echo "Creating conda environment (sam). Please make sure you have conda installed ..."
conda create --name sam --file spec_file.txt

echo "Activating the conda environment (sam)" 
conda activate sam

echo "Installing some extra dependencies through pip ..."
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

echo "Downloading the SAM checkpoint from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
echo "If this doesn\'t happen automatically, please download checkpoint yourself and put it in checkpoints"
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
mv sam_vit_h_4b8939.pth checkpoints/

echo "Downloading checkpoint from https://drive.google.com/uc?id=1WZc7wen-K2EhwmZaRUebK76nHWtKPlk_"
echo "If this doesn\'t happen automatically, please download checkpoint through browser and put it in lightning_logs/version_27/checkpoints/"
echo "It\'s a 2.6 GB checkpoint!" 
gdown https://drive.google.com/uc?id=1WZc7wen-K2EhwmZaRUebK76nHWtKPlk_
mv epoch=40-step=55432.ckpt lightning_logs/version_27/checkpoints/

echo "Running app ..." 
python3 app_int.py
