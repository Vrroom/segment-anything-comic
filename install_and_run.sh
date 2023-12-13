echo "Creating conda environment (sam). Please make sure you have conda installed ..."
conda create --name sam --file spec_file.txt

echo "Activating the conda environment (sam)" 
conda activate sam

echo "Installing some extra dependencies through pip ..."
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

echo "Downloading checkpoint from https://drive.google.com/uc?id=1WZc7wen-K2EhwmZaRUebK76nHWtKPlk_"
echo "If this doesn\'t happen automatically, please download checkpoint through browser and put it in lightning_logs/version_27/checkpoints/"
echo "It\'s a 2.6 GB checkpoint!" 
gdown https://drive.google.com/uc?id=1WZc7wen-K2EhwmZaRUebK76nHWtKPlk_
mv epoch=40-step=55432.ckpt lightning_logs/version_27/checkpoints/

echo "Running app ..." 
python3 app_int.py
