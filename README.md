## Building from source (with CMake)

1. Install PHAST and note its install prefix, e.g. `/usr/local` or `/opt/homebrew/opt/phast`.
2. Configure and build:

   ```bash
   cmake -S . -B build \
         -DCMAKE_BUILD_TYPE=Release \
         -DPHAST_ROOT=/opt/homebrew/opt/phast
   cmake --build build
   cmake --install build --prefix /usr/local
