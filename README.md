BGPsyche
================================================================================

BGPsyche is a system to predict BGP paths between arbitrary autonomous systems
(ASes) using deep learning. It was written as part of my (@dominiksta) [masters
thesis](2024_Stahmer_BGPsyche.pdf). It should be considered a prototype
implementation, not a finished product. Nevertheless, running (and training)
BGPsyche should be relatively straightforward and is outlined in this README.

BGPsyche is built for *both* [PyPy](https://www.pypy.org/download.html) *and*
CPython 3.10 on Ubuntu 22.04. Other OS/Python combinations may or may not work.

System Dependencies
--------------------------------------------------------------------------------

- PyPy 3.10 and CPython 3.10
- A [Rust toolchain](https://www.rust-lang.org/tools/install): BGPsyche uses the
  [pybgpkit-parser](https://github.com/bgpkit) MRT parser, which is written in
  Rust. The download and compilation of this module will be done automatically
  by pip.
  
Install and Run
--------------------------------------------------------------------------------
  
Start by creating an `env.ini` file in `/bgspyche`, following the template
`env.ini.template`. You will have to create accounts for both PeeringDB and
MaxMind GeoLite2. Then run the following commands:

```bash
pypy3.10 -m venv .venv-pypy
. .venv-pypy/bin/activate
pip install -r bgpsyche/requirements.pypy.txt
deactivate

python3.10 -m venv .venv
. .venv/bin/activate
pip install -r bgpsyche/requirements.main.txt

# this can take up to half an hour on the initial sync
python -m bgpsyche peeringdb_sync

python -m bgpsyche --help
```

Usage
--------------------------------------------------------------------------------

TODO

When BGPsyche runs for the first time, it will have to pull down quite a few
datasets from the internet and compute various features from those. Please just
be patient here. There may also be cases where you will run out of memory on the
first run and have to just run the same command again. As stated, its a prototype
implementation. All datasets and computed features are cached on disk though, so
re-running the same command will be much quicker and should not cause any issues.

### Using the pre-trained ("silver") model

BGPsyche offers a pre-trained model in the repository. This model was trained
with what was found to be the ideal parameters and features in the
[thesis](2024_Stahmer_BGPsyche.pdf). It was therefore called the "silver" model.

Usage of a pre-trained model requires about 6 GB of RAM. Almost all of the
memory is taken up by various datastructures, not the model itself. Running a
larger model should therefore take a comparable amount of memory.

TODO date of data (not just 23-05-1)

BGPsyche exposes an HTTP API with a single endpoint for path predictions:

```bash
# in one terminal
python -m bgpsyche listen 8080
# in another terminal
curl "http://localhost:8080/predict?source=3320&sink=6939"
```

### Training

Training a model with default configuration (another "silver" model) takes about
16GB of RAM and 16GB of VRAM. Please see `make_dataset.py` for adjusting the
dataset size if necessary. In general, if you want to train your own model you
probably want to read (or at least skim) the
[thesis](2024_Stahmer_BGPsyche.pdf). The most relevant things to adjust are
arguably in `vectorize_features.py`, `enrich.py`, `make_dataset.py` and
`classifier_nn.py`.

```bash
# to evaluate the model by comparing to real paths 
# (with output tensorboard for visualization)
python -m bgpsyche train_and_evaluate

# in another terminal
python -m bgpsyche tensorboard
```