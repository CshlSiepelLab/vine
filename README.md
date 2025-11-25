## Building from source (with CMake)

1. Install PHAST and note its install prefix, e.g. `/usr/local` or `/opt/homebrew/opt/phast`.
2. Configure and build:

   ```bash
   cmake -S . -B build \
         -DCMAKE_BUILD_TYPE=Release \
         -DPHAST_ROOT=/opt/homebrew/opt/phast
   cmake --build build
   cmake --install build --prefix /usr/local

## License

VINE is distributed under the **BSD 3-Clause License**, a permissive academic
license that allows redistribution and modification provided that attribution
is retained.

- You are free to use, modify, and redistribute VINE in source or binary form.
- You must retain the copyright notice and license terms in any redistribution.
- The name of the authors may not be used to endorse derived products without
  permission.

See the file [`LICENSE`] for the full license text.

VINE depends on the **PHAST** library, which is also distributed under a BSD
License.
