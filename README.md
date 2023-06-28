# Emotional Text to Speech
Official implementation of EmoSpeech, accepted to SSW12


## How to run

### Build env

You can build an environment with `Docker` or `Conda`.
#### To set up environment with Docker

If you don't have Docker installed, please follow the links to find installation instructions for [Ubuntu](https://docs.docker.com/desktop/install/linux-install/), [Mac](https://docs.docker.com/desktop/install/mac-install/) or [Windows](https://docs.docker.com/desktop/install/windows-install/).

Build docker image:

    docker build -t emospeech .

Run docker image:

    bash run_docker.sh
      
#### To set up environment with Conda
If you don't have Conda installed,  please find the installation instructions for your OS [here](https://docs.conda.io/en/latest/miniconda.html).

      conda create -n etts python=3.10
      conda activate etts
      pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
      pip install -r requirements.txt

If you have different version of cuda on your machine you can find applicable link for pytorch installation [here](https://pytorch.org/get-started/previous-versions/).


### Download and preprocess data
We used data of 10 English Speakers from [ESD dataset](https://github.com/HLTSingapore/Emotional-Speech-Data). To download all `.wav`, `.txt` files along with `.TextGrid` files created using [MFA](https://github.com/MontrealCorpusTools/mfa-models):

      bash download_data.sh
 
To train a model we need precomputed durations, energy, pitch and eGeMap features. From `src` directory run:

      python -m src.preprocess.preprocess
      
This is how your data folder should look like:


      .
      ├── data
      │   ├── ssw_esd
      │   ├── test_ids.txt
      │   ├── val_ids.txt
      └── └── preprocessed
              ├── duration
              ├── egemap
              ├── energy
              ├── mel
              ├── phones.json
              ├── pitch
              ├── stats.json
              ├── test.txt
              ├── train.txt
              ├── trimmed_wav
              └── val.txt
        
### Training
1. Configure arguments in `config/config.py`
2. Run `python -m src.scripts.train `

### Testing
Testing is implemented on testing subset of ESD dataset. To synthesize audio and compute neural MOS (NISQA TTS):
1. Configure arguments in `config/config.py` under `# Inference` section
2. Run `python -m src.scripts.test`

You can find NISQA TTS for original, reconstructed and generated audio in `test.log`.

### Inference
EmoSpeech is trained on phoneme sequences. Supported phones can be found in  `data/preprocessed/phones.json`. This repositroy is created for academic research and doesn't support automatic grapheme-to-phoneme conversion. However, if you would like to synthesize arbitrary sentence with emotion conditioning you can:
1. Generate phoneme sequence from graphemes with [MFA](https://github.com/MontrealCorpusTools/mfa-models).
      1.1 Follow the [installation guide](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html)
      
      1.2 Download english g2p model: `mfa model download g2p english_us_arpa`
      
      1.3 Generate phoneme.txt from graphemes.txt: `mfa g2p graphemes.txt english_us_arpa phoneme.txt`
      
2. Run `python -m src.scripts.inference`, specifying phone sequence with `--sq ` argument. For example


 ``` 
 python -m src.scripts.inference --sq "S P IY2 K ER1 F AY1 V  T AO1 K IH0 NG W IH0 TH AE1 NG G R IY0 IH0 M OW0 SH AH0 N"
 ```
 
 If result file is not synthesied, check `inference.log` for OOV phones. 
                
                
## References
1. [FastSpeech 2 - PyTorch Implementation](https://github.com/ming024/FastSpeech2)
2. [iSTFTNet : Fast and Lightweight Mel-spectrogram Vocoder Incorporating Inverse Short-time Fourier Transform](https://github.com/rishikksh20/iSTFTNet-pytorch)
3. [Publicly Available Emotional Speech Dataset (ESD) for Speech Synthesis and Voice Conversion](https://github.com/HLTSingapore/Emotional-Speech-Data)
4. [NISQA: Speech Quality and Naturalness Assessment](https://github.com/gabrielmittag/NISQA)
5. [Montreal Forced Aligner Models](https://github.com/MontrealCorpusTools/mfa-models)
6. [Modified VocGAN](https://github.com/rishikksh20/VocGAN/tree/master)
7. [AdaSpeech](https://github.com/rishikksh20/AdaSpeech)
