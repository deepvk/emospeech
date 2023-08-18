#!/bin/bash
cd /app
mkdir data
cd data
gdown https://drive.google.com/uc?id=1n29y5hr8JK2tGEdYNLtIXO8hWw7DkdJ1; # download ssw_esd.zip (wavs, txt, TextGrids)
gdown https://drive.google.com/uc?id=1F-_o_6x43IuL-81QHozAab7l3PPG2DNE; # download val_ids.txt
gdown https://drive.google.com/uc?id=1VgvuU1p79GLPikwGl3lO6lanDdneHZud; # download test_ids.txt
gdown https://drive.google.com/uc?id=1KqPK3A8JpB57pzy5dEgkX9RRRkkPZS7S; # download vocoder_checkpoint
gdown https://drive.google.com/uc?id=1SHInZhoiv6yraJqPL-J5XJ4_aRFOCxR3; # download EmoSpeech checkpoint
gdown https://drive.google.com/uc?id=1_g23gaPC7K1hO-kWJk9JOPJZR8olimxw; # download phones.json
unzip -qq ssw_esd.zip
