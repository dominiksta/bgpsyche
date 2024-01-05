# BGPsyche

## Development

BGPsyche is built for *both* [PyPy](https://www.pypy.org/download.html) *and*
CPython 3.10 on Ubuntu 22.04. Other OS/Python combinations may or may not work.

### System Dependencies

- PyPy 3.10 and CPython 3.10
- A [Rust toolchain](https://www.rust-lang.org/tools/install): BGPsyche uses the
  [pybgpkit-parser](https://github.com/bgpkit) MRT parser, which is written in
  Rust.
  
### Install and Run
  
Start by creating an `env.ini` file in `/bgspyche`, following the template
`env.ini.template`. Then run the following commands:

```{bash}
pypy3.10 -m venv .venv-pypy
. .venv-pypy/bin/activate
pip install -r bgpsyche/requirements.pypy.txt
deactivate

python3.10 -m venv .venv
. .venv/bin/activate
pip install -r bgpsyche/requirements.main.txt

python -m bgpsyche --help
```

## TODO: `mrt_tier1` and `mrx_ixp`?