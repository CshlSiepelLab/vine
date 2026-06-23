# VINE

**VINE (Variational Inference with Node Embeddings)** is a program and set
of supporting libraries for variational inference of phylogenetic trees.

## Requirements

- CMake 3.x or later
- C/C++ compiler (GCC, Clang, etc.)
- [PHAST](https://github.com/CshlSiepelLab/phast) (Phylogenetic Analysis with Space/Time models)

## Installation

VINE is easily installable via Bioconda:

```bash
conda install -c conda-forge -c bioconda vine-phylo
```

or Homebrew:

```bash
brew tap CshlSiepelLab/tools
brew install vine
```

In both cases, PHAST will be installed as a dependency if needed.

## Building from Source

If PHAST is installed in a standard location, CMake will usually find it
automatically. Otherwise, you can specify its install prefix explicitly:

```bash
cmake -S . -B build \
   -DCMAKE_BUILD_TYPE=Release \
   -DPHAST_ROOT=/path/to/phast
cmake --build build
cmake --install build
```

Here, `PHAST_ROOT` should point to the installation prefix of PHAST (e.g.,
`/opt/homebrew/opt/phast`).

## Multithreading Support
Uses OpenMP by default if found by CMake. OpenMP is not required. 

Note for MacOS: if OpenMP is not found by default, try explicitly
specifying the LLVM compiler during configuration, e.g., 
```sh
cmake -S . -B build -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang
```

The number of threads can be controlled with the -j option.

## Usage

For detailed usage of the main vine executable and supporting programs, run them
with `--help`:

```bash
vine --help
```

The same full usage message for vine is also available
directly in the source tree:
[vine.help_src](src/progs/vine.help_src).

## Support

For questions or bug reports, please use the [GitHub issue tracker](https://github.com/CshlSiepelLab/vine/issues).

## License

Both PHAST and VINE are distributed under the BSD 3-Clause License, a
permissive academic license that allows redistribution and modification
provided that attribution is retained. See [LICENSE](LICENSE) for details.
