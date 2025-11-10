Faster R-CNN with a ResNet-50 backbone + Feature Pyramid Network (FPN)

📁 Dataset Contents

Each image (road scene) is annotated with bounding boxes for 4 main classes:

Label	Meaning	Description
D00	Longitudinal crack	Crack along the direction of travel
D10	Transverse crack	Crack across the road
D20	Alligator crack	Web-like surface crack
D40	Pothole	Visible depression or hole




#######################################
# 🚧 Pothole Spotter AI
#######################################
# Machine learning project that detects potholes 
# from images or live video streams using PyTorch.
#######################################

#######################################
# 🔧 Setup & Usage
#######################################


# ASSUMING YOU HAVE PYTHON INSTALLED IF NOT DOWNLOAD IT FROM THIS LINK AND INSTALL IT 
https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe


# 1️⃣ Clone the repository
git clone https://github.com/NicholasTechmoverai/undisputedPothholeSpotterAi.git
cd potholeSpotterAI

# 2️⃣ Create a virtual environment
# 👉 Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate

# 👉 Linux / macOS
python -m venv .venv
source .venv/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt


# 4️⃣ download the model
download the trained model from the link below and add it to the project main folder
https://drive.google.com/file/d/1o1B5cTTnR9FsGYgdYZtFEc7Dje8qSK9N/view?usp=sharing

# 5 Run the application
python app.py

#######################################
# 📝 Notes
#######################################
# - Use Python 3.8+ 
# - Always activate your .venv before running
# - Update dependencies if needed:
pip freeze > requirements.txt
