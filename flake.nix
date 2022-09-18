{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-22.05";
    online_lib = {
        # this is a personal mirror of the EigenPositIntegration. For some reason, I cannot access the private github repo even with a token. So now I have a personal gitea instance on which this is hosted "public"
        url = "git+http://10.0.0.1:3000/aethan/EigenPositIntegration.git";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, online_lib, flake-utils }:
    let
      # wrap this in another let .. in to add the hydra job only for a single architecture
      output_set = flake-utils.lib.eachDefaultSystem (system:
        let
            pkgs = nixpkgs.legacyPackages.${system};
        in
        rec {
            packages = flake-utils.lib.flattenTree {
                solvers = pkgs.gcc10Stdenv.mkDerivation {
                    name = "CustomKrylovSolvers";
                    src = ./.;

                    nativeBuildInputs = [pkgs.cmake];

                    buildInputs = [
                        online_lib.packages.${system}.eigen
                        online_lib.packages.${system}.universal
                        online_lib.packages.${system}.eigen_universal_integration
                        pkgs.llvmPackages.openmp
                    ];

                    checkPhase = ''
                        ctest
                    '';

                    cmakeFlags = [
                        "-DTESTS=ON"
                    ];

                    doCheck = true;
                };
            };

            defaultPackage = packages.solvers;

            devShell = pkgs.mkShell {
                buildInputs = [
                    online_lib.packages.${system}.eigen
                    online_lib.packages.${system}.universal
                    online_lib.packages.${system}.eigen_universal_integration
                    pkgs.cmake
                    # pkgs.gdbgui
                ];

                shellHook = ''
                    function run_cmake_build() {
                        cmake -B /mnt/RamDisk/build -DDEBUG=ON
                        cmake --build /mnt/RamDisk/build -j12
                    }

                    function build_and_run() {
                        run_cmake_build
                        /mnt/RamDisk/build/ShermanLoad
                    }
                    function build_and_test() {
                        run_cmake_build
                        pushd /mnt/RamDisk/build
                        ctest
                        popd
                    }
                '';
            };


        }
    );
    in
        output_set // { hydraJobs.build."aarch64-linux" = output_set.defaultPackage."aarch64-linux"; };
    }
