{
  description = "NInfer — C++/CUDA inference engine and server for .ninfer model artifacts";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # Strata (github.com/Niko1221/Strata) has no flake — fetch the source tree
    # (header-only GGUF reader, CUDA kernels, ggml block-layout headers) and
    # pin it to a commit for reproducibility.
    strata = {
      url = "github:Niko1221/Strata/6f32ec070f23ced9f50e704d854d775da52591ab";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, strata }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      cuda = pkgs.cudaPackages_13_2;

      # Prebuilt llama.cpp webui (ggml-org/llama-ui), pinned to a specific build
      # rather than the rolling "latest" pointer for reproducibility. A single
      # prebuilt dist.tar.gz is published per build; its entries are ./-prefixed,
      # so extracting it directly yields the servable tree (index.html at root).
      llamaWebui = pkgs.stdenvNoCC.mkDerivation rec {
        pname = "llama-ui-webui";
        version = "b10021";

        src = pkgs.fetchurl {
          url = "https://huggingface.co/buckets/ggml-org/llama-ui/resolve/${version}/dist.tar.gz";
          # Matches the upstream-published dist.tar.gz.sha256 sidecar.
          sha256 = "7726ec9d4b7fe63536f70dcd14813a4054819317ba4df78539ea2bfc5aa7db2a";
        };

        # The prebuilt archive is the whole deliverable: extract it straight into
        # $out and skip the (vacuous) build/install phases.
        dontConfigure = true;
        dontBuild = true;
        dontCheck = true;
        unpackPhase = ''
          mkdir -p "$out/share/ninfer/webui"
          tar xzf "$src" -C "$out/share/ninfer/webui"
        '';
        buildPhase = ":";
        installPhase = ":";

        meta = with pkgs.lib; {
          description = "Prebuilt llama.cpp webui (ggml-org/llama-ui), pinned build ${version}";
        };
      };

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
          llamaWebui
        ];

        # CMake also bundles the tree into the build dir (build/share/ninfer/webui) for
        # dev builds; the flake points NINFER_WEBUI_SRC at the fetched webui tree.
        cmakeFlags = [
          "-DCMAKE_CUDA_ARCHITECTURES=120a"
          "-DNINFER_WEBUI_SRC=${llamaWebui}/share/ninfer/webui"
          # The fetched Strata source tree (GGUF reader + CUDA kernels).
          "-DNINFER_STRATA_SRC=${strata}"
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
          # Bundle the prebuilt webui so --webui can serve it from
          # <exe-dir>/../share/ninfer/webui with no runtime download.
          install -d "$out/share/ninfer/webui"
          cp -a "${llamaWebui}/share/ninfer/webui/." "$out/share/ninfer/webui/"
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
        llamaWebui = llamaWebui;
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

        # Expose the fetched Strata source tree to out-of-flake dev builds in
        # /tmp/ninfer-build (cmake -DNINFER_STRATA_SRC=$NINFER_STRATA_SRC).
        env.NINFER_STRATA_SRC = "${strata}";
      };
    };
}
