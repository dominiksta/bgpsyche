# BGPsyche

## Development

BGPsyche is built for [PyPy](https://www.pypy.org/download.html) 3.10 on Ubuntu
22.04. Other OS/Python combinations may or may not work.

You may also have to install a [Rust
toolchain](https://www.rust-lang.org/tools/install), because BGPsyche uses the
[pybgpkit-parser](https://github.com/bgpkit) MRT parser, which is written in
Rust.

You will also need to create a `env.ini` file in `/bgspyche`, following the
template `env.ini.template`.

```
pypy3.10 -m venv .venv
. .venv/bin/activate
pip install -r bgpsyche/requirements.txt

python -m bgpsyche --help
```

## TODO: `mrt_tier1` and `mrx_ixp`?