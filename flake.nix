{
  description = "NInfer — C++/CUDA inference engine and server for .ninfer model artifacts";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      cuda = pkgs.cudaPackages_13_2;

      # CMake 4.x + Ninja: the Clang CXX compiler module (Clang-CXX.cmake) sets
      # CMAKE_CXX_SCANDEP_SOURCE *unconditionally* for Clang >= 16 with the GNU
      # frontend, so the Ninja generator emits a clang-scan-deps `.ddi` (Dynamic
      # Dependency Index) rule for every C++ translation unit. In the Nix sandbox
      # that scanner tool is either -NOTFOUND (Clang-FindBinUtils.cmake locates it
      # via HINTS=<compiler bin dir> with NO_CMAKE_ENVIRONMENT_PATH, so stripping
      # PATH cannot help) or, when found, is the unwrapped pkgs.clang scanner that
      # lacks the -isystem include paths the Nix clang wrapper injects, so the
      # scan cannot find the C++ standard headers (cstddef, cstdint, ...) and
      # every C++ TU fails. This snippet is fed to CMake via CMAKE_PROJECT_INCLUDE,
      # which is included as the last step of project() — i.e. after the compiler
      # modules have run — clearing the variable so the Ninja generator drops the
      # `.ddi` rule and falls back to compiler-based dependency generation (-MD
      # -MF), which works because the clang wrapper carries the correct include
      # paths. (CMAKE_PROJECT_TOP_LEVEL_INCLUDES would be too early: it runs before
      # any language is enabled, so the variable would not yet be set.)
      scandepsDisable = pkgs.writeText "ninfer-disable-scandeps.cmake" ''
        unset(CMAKE_CXX_SCANDEP_SOURCE)
      '';

      # The clangStdenv wrapper compiles C++ against its libcpp headers
      # (gcc-16.2.0) while the stdenv runtime on the NIX_LDFLAGS -L paths is
      # gcc-15.3.0. The monolithic 16.2.0 gcc is a langJit build with no `lib`
      # output, so the wrapper's libcxx-ldflags is empty and nothing injects
      # the matching runtime: code that pulls non-inline libstdc++ symbols
      # (e.g. std::counting_semaphore in the vendored xgrammar grammar
      # compiler) fails to link against the 15.3.0 library. Link the 16.2.0
      # libstdc++ (the NG split-package build) explicitly; the absolute path
      # makes it win over the 15.3.0 -L search order.
      libstdcxx = pkgs.gccNGPackages_16.libstdcxx;

      ninfer = pkgs.clangStdenv.mkDerivation rec {
        pname = "ninfer";
        version = "0.1.0";

        src = pkgs.lib.cleanSource ./.;

        nativeBuildInputs = [
          pkgs.cmake
          pkgs.ninja
          pkgs.pkg-config
          cuda.cuda_nvcc
        ];

        buildInputs = [
          pkgs.ffmpeg
          pkgs.curl.dev
          cuda.cuda_cudart
          cuda.cuda_nvtx
          # Matching C++ runtime for the wrapper's gcc-16.2.0 headers (see the
          # libstdcxx binding above); also a runtime dependency of the binaries.
          libstdcxx
          # Nixpkgs-provided libraries that were previously vendored under third_party/.
          pkgs.httplib
          pkgs.nlohmann_json
          (pkgs.spdlog.override { staticBuild = true; })
          pkgs.utf8proc
        ];

        cmakeFlags = [
          "-DCMAKE_CUDA_ARCHITECTURES=120a"
          # Opt out of the Ninja clang-scan-deps `.ddi` rule (see the comment on
          # the scandepsDisable store path): with CMAKE_CXX_SCANDEP_SOURCE unset
          # the Ninja generator falls back to compiler-based dependency
          # generation, which works in the Nix sandbox.
          "-DCMAKE_PROJECT_INCLUDE=${scandepsDisable}"
          # Explicit C++ standard library: CMake leaves CMAKE_CXX_STANDARD_LIBRARIES
          # empty for Clang (driver auto-link), and the wrapper's empty
          # libcxx-ldflags means nothing else adds libstdc++ — so the 16.2.0
          # runtime is linked by absolute path (beats the 15.3.0 -L order).
          "-DCMAKE_CXX_STANDARD_LIBRARIES=${libstdcxx}/lib/libstdc++.so"
          # Embed the 16.2.0 libstdc++ as DT_RPATH (old dtags => searched before
          # LD_LIBRARY_PATH): the runtime environment resolves libstdc++.so.6 to
          # the stdenv's 15.3.0 library, which lacks the GLIBCXX_3.4.35 nodes
          # these binaries require.
          "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--disable-new-dtags,-rpath,${libstdcxx}/lib"
        ];

        enableParallelBuilding = true;
        doCheck = false;

        installPhase = ''
          runHook preInstall
          install -Dm0755 apps/ninfer "$out/bin/ninfer"
          install -Dm0755 apps/ninfer-serve "$out/bin/ninfer-serve"
          runHook postInstall
        '';

        passthru = {
          inherit cuda;
        };


      };
    in
    {
      packages.${system} = {
        default = ninfer;
        ninfer = ninfer;
      };

      apps.${system} = {
        default = {
          type = "app";
          program = "${ninfer}/bin/ninfer";
        };
        serve = {
          type = "app";
          program = "${ninfer}/bin/ninfer-serve";
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          pkgs.cmake
          pkgs.ninja
          pkgs.pkg-config
          pkgs.clang-tools
          cuda.cuda_nvcc
          pkgs.ffmpeg
          pkgs.curl
          cuda.cuda_cudart
          cuda.cuda_nvtx
          # cuda_profiler_api.h (cudaProfilerStart/Stop) for the bench target;
          # its default output is the CUPTI include tree.
          cuda.cuda_profiler_api
          # Nixpkgs-provided libraries that were previously vendored under third_party/.
          pkgs.httplib
          pkgs.nlohmann_json
          (pkgs.spdlog.override { staticBuild = true; })
          pkgs.utf8proc
        ];
      };
    };
}
