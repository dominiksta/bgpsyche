# BGPsyche

## Development

BGPSyche is built for [PyPy](https://www.pypy.org/download.html) 3.9 on Ubuntu
22.04. Other OS/Python combinations may or may not work.

You may also have to install a [rust
toolchain](https://www.rust-lang.org/tools/install), because BGPSyche uses the
[pybgpkit-parser](https://github.com/bgpkit) MRT parser, which is written in
rust.

You will also need to create a `env.ini` file in `/bgspyche`, following the
template `env.ini.template`.

```
pypy3.9 -m venv .venv
. .venv/bin/activate
pip install -r bgpsyche/requirements.txt

python -m bgpsyche --help
```