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
      cuda = pkgs.cudaPackages_13;

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
        ];

        cmakeFlags = [ "-DCMAKE_CUDA_ARCHITECTURES=120a" ];

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
        ];
      };
    };
}