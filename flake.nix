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

      ninfer = pkgs.clangStdenv.mkDerivation rec {
        pname = "ninfer";
        version = "0.1.0";

        src = pkgs.lib.cleanSource ./.;

        nativeBuildInputs = [
          pkgs.cmake
          pkgs.ninja
          pkgs.pkg-config
          pkgs.clang-tools
          cuda.cuda_nvcc
        ];

        buildInputs = [
          pkgs.ffmpeg
          pkgs.curl.dev
          cuda.cuda_cudart
          cuda.cuda_nvtx
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
          # Nixpkgs-provided libraries that were previously vendored under third_party/.
          pkgs.httplib
          pkgs.nlohmann_json
          (pkgs.spdlog.override { staticBuild = true; })
          pkgs.utf8proc
        ];
      };
    };
}
