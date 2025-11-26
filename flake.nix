{
  description = "Event-based detection challenge environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    fenix = {
      url = "https://flakehub.com/f/nix-community/fenix/0.1.tar.gz";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, fenix, ... }@inputs:
    let
      supportedSystems =
        [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f:
        nixpkgs.lib.genAttrs supportedSystems (system:
          f {
            pkgs = import nixpkgs {
              inherit system;
              overlays = [ self.overlays.default ];
              config.allowUnfree = true;
            };
          });
    in {
      overlays.default = final: prev: {
        rustToolchain = with fenix.packages.${prev.stdenv.hostPlatform.system};
          combine (with latest; [ clippy rustc cargo rustfmt rust-src ]);
      };

      devShells = forEachSupportedSystem ({ pkgs }:
        let
          commonDeps = with pkgs; [ rustToolchain pkg-config ];

          linuxDeps = with pkgs; [
            alsa-lib
            udev
            libxkbcommon
            wayland
            wayland-protocols
            vulkan-loader
            xorg.libX11
            xorg.libXcursor
            xorg.libXi
            xorg.libXrandr
          ];

          darwinDeps = if pkgs.stdenv.isDarwin then
            with pkgs.darwin.apple_sdk.frameworks;
            [
              AppKit
              CoreFoundation
              CoreGraphics
              CoreVideo
              Foundation
              IOKit
              Metal
              QuartzCore
              Security
              SystemConfiguration
            ] ++ [ pkgs.libiconv ]
          else
            [ ];
        in {
          default = pkgs.mkShell {
            packages = commonDeps
              ++ (if pkgs.stdenv.isLinux then linuxDeps else [ ]) ++ darwinDeps;

            env = {
              RUST_SRC_PATH =
                "${pkgs.rustToolchain}/lib/rustlib/src/rust/library";
              LD_LIBRARY_PATH = if pkgs.stdenv.isLinux then
                "${pkgs.lib.makeLibraryPath linuxDeps}:$LD_LIBRARY_PATH"
              else
                "";
            };

            shellHook = ''
              echo "Event-based detection environment loaded"
              echo "Rust: $(rustc --version)"
            '';
          };
        });

      # Runnable apps for common workflows
      # NOTE: These scripts assume you run them from the project root directory
      apps = forEachSupportedSystem ({ pkgs }:
        let
          linuxDeps = with pkgs; [
            alsa-lib
            udev
            libxkbcommon
            wayland
            wayland-protocols
            vulkan-loader
            xorg.libX11
            xorg.libXcursor
            xorg.libXi
            xorg.libXrandr
          ];
          ldPath = pkgs.lib.makeLibraryPath linuxDeps;
        in {
          # Generate synthetic test data with ground truth
          generate_data = {
            type = "app";
            program = toString (pkgs.writeShellScript "generate_data" ''
              export LD_LIBRARY_PATH="${ldPath}:''${LD_LIBRARY_PATH:-}"
              exec cargo run --release --bin generate_synthetic
            '');
          };

          # Run hyperparameter optimization on synthetic data
          optimise_params = {
            type = "app";
            program = toString (pkgs.writeShellScript "optimise_params" ''
              export LD_LIBRARY_PATH="${ldPath}:''${LD_LIBRARY_PATH:-}"
              exec cargo run --release --bin hypersearch -- \
                --data data/synthetic/fan_test.dat \
                --output results/synthetic_search.csv \
                --window-sizes 50,100,200,500 \
                --thresholds 25,50,100,200,500 \
                --frames 30
            '');
          };

          # Live comparison of all detectors (synthetic data)
          compare_live = {
            type = "app";
            program = toString (pkgs.writeShellScript "compare_live" ''
              export LD_LIBRARY_PATH="${ldPath}:''${LD_LIBRARY_PATH:-}"
              exec cargo run --release --bin compare_live -- \
                --config config/detectors.toml \
                data/synthetic/fan_test.dat
            '');
          };

          # Live comparison with real fan data
          compare_real = {
            type = "app";
            program = toString (pkgs.writeShellScript "compare_real" ''
              export LD_LIBRARY_PATH="${ldPath}:''${LD_LIBRARY_PATH:-}"
              exec cargo run --release --bin compare_live -- \
                --config config/detectors.toml \
                data/fan/fan_const_rpm.dat
            '');
          };

          # Main visualizer (pass data file as argument)
          visualize = {
            type = "app";
            program = toString (pkgs.writeShellScript "visualize" ''
              export LD_LIBRARY_PATH="${ldPath}:''${LD_LIBRARY_PATH:-}"
              exec cargo run --release -- "''${1:-data/fan/fan_const_rpm.dat}"
            '');
          };
        });
    };
}
