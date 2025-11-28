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
        in
        let
          ldPath = pkgs.lib.makeLibraryPath linuxDeps;
          # Wrapper scripts that can be added to PATH
          generate_data_script = pkgs.writeShellScriptBin "generate_data" ''
            export LD_LIBRARY_PATH="${ldPath}:''${LD_LIBRARY_PATH:-}"
            exec cargo run --release --bin generate_synthetic "$@"
          '';
          optimise_params_script = pkgs.writeShellScriptBin "optimise_params" ''
            export LD_LIBRARY_PATH="${ldPath}:''${LD_LIBRARY_PATH:-}"
            exec cargo run --release --bin hypersearch -- \
              --data data/synthetic/fan_test.dat \
              --output results/synthetic_search.csv \
              --window-sizes 500,1000,2000,5000 \
              --thresholds 0.5,1.0,2.0,4.0 \
              --frames 30 "$@"
          '';
          compare_live_script = pkgs.writeShellScriptBin "compare_live" ''
            export LD_LIBRARY_PATH="${ldPath}:''${LD_LIBRARY_PATH:-}"
            exec cargo run --release --bin compare_live -- \
              --config config/detectors.toml \
              data/synthetic/fan_test.dat "$@"
          '';
          compare_real_script = pkgs.writeShellScriptBin "compare_real" ''
            export LD_LIBRARY_PATH="${ldPath}:''${LD_LIBRARY_PATH:-}"
            exec cargo run --release --bin compare_live -- \
              --config config/detectors.toml \
              data/fan/fan_const_rpm.dat "$@"
          '';
          visualize_script = pkgs.writeShellScriptBin "visualize" ''
            export LD_LIBRARY_PATH="${ldPath}:''${LD_LIBRARY_PATH:-}"
            exec cargo run --release -- "''${1:-data/fan/fan_const_rpm.dat}"
          '';
          evaluate_quick_script = pkgs.writeShellScriptBin "evaluate_quick" ''
            export LD_LIBRARY_PATH="${ldPath}:''${LD_LIBRARY_PATH:-}"
            exec cargo run --release --bin evaluate_cmax_slam -- \
              --data data/synthetic/fan_test.dat \
              --output results/cmax_quick.csv "$@"
          '';
          evaluate_sweep_script = pkgs.writeShellScriptBin "evaluate_sweep" ''
            export LD_LIBRARY_PATH="${ldPath}:''${LD_LIBRARY_PATH:-}"

            echo "Generating test datasets..."
            for rpm in 600 900 1200 1500 2000; do
              for blades in 2 3 4 5; do
                cargo run --release --bin generate_synthetic -- \
                  --rpm $rpm --blades $blades \
                  --output "data/synthetic/sweep_rpm''${rpm}_b''${blades}.dat"
              done
            done

            echo "Running evaluation..."
            rm -f results/cmax_sweep.csv
            for f in data/synthetic/sweep_*.dat; do
              cargo run --release --bin evaluate_cmax_slam -- \
                --data "$f" \
                --output results/cmax_sweep.csv "$@"
            done

            echo "Results written to results/cmax_sweep.csv"
          '';
          workflowScripts = if pkgs.stdenv.isLinux then [
            generate_data_script
            optimise_params_script
            compare_live_script
            compare_real_script
            visualize_script
            evaluate_quick_script
            evaluate_sweep_script
          ] else [ ];
        in {
          default = pkgs.mkShell {
            packages = commonDeps
              ++ (if pkgs.stdenv.isLinux then linuxDeps else [ ])
              ++ darwinDeps
              ++ workflowScripts;

            env = {
              RUST_SRC_PATH =
                "${pkgs.rustToolchain}/lib/rustlib/src/rust/library";
              LD_LIBRARY_PATH = if pkgs.stdenv.isLinux then
                "${ldPath}:$LD_LIBRARY_PATH"
              else
                "";
            };

            shellHook = ''
              echo "Event-based detection environment loaded"
              echo "Rust: $(rustc --version)"
              echo ""
              echo "Available commands:"
              echo "  generate_data   - Generate synthetic test data"
              echo "  optimise_params - Run hyperparameter optimization"
              echo "  compare_live    - Compare detectors (synthetic)"
              echo "  compare_real    - Compare detectors (real fan data)"
              echo "  visualize       - Main visualizer"
              echo "  evaluate_quick  - Quick single-dataset evaluation"
              echo "  evaluate_sweep  - Full parameter sweep evaluation"
            '';
          };
        });

      # Runnable apps for common workflows
      # NOTE: These scripts assume you run them from the project root directory
      apps = forEachSupportedSystem ({ pkgs }:
        let
          # Runtime libraries (for LD_LIBRARY_PATH)
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
          # Dev packages with .pc files (for PKG_CONFIG_PATH)
          # Many nix packages split dev files into .dev output
          linuxDevDeps = with pkgs; [
            alsa-lib.dev
            udev.dev
            libxkbcommon.dev
            wayland.dev
            wayland-protocols
            vulkan-loader.dev
            xorg.libX11.dev
            xorg.libXcursor.dev
            xorg.libXi.dev
            xorg.libXrandr.dev
          ];
          ldPath = pkgs.lib.makeLibraryPath linuxDeps;
          pkgConfigPath = pkgs.lib.makeSearchPath "lib/pkgconfig" linuxDevDeps
            + ":" + pkgs.lib.makeSearchPath "share/pkgconfig" linuxDevDeps;
          cargo = pkgs.rustToolchain;
          cc = pkgs.stdenv.cc;
          # Common environment setup for all apps
          envSetup = ''
            export LD_LIBRARY_PATH="${ldPath}:''${LD_LIBRARY_PATH:-}"
            export PKG_CONFIG_PATH="${pkgConfigPath}:''${PKG_CONFIG_PATH:-}"
            export PATH="${cargo}/bin:${pkgs.pkg-config}/bin:${cc}/bin:$PATH"
          '';
        in {
          # Generate synthetic test data with ground truth
          generate_data = {
            type = "app";
            program = toString (pkgs.writeShellScript "generate_data" ''
              ${envSetup}
              exec cargo run --release --bin generate_synthetic
            '');
          };

          # Run hyperparameter optimization on synthetic data
          optimise_params = {
            type = "app";
            program = toString (pkgs.writeShellScript "optimise_params" ''
              ${envSetup}
              exec cargo run --release --bin hypersearch -- \
                --data data/synthetic/fan_test.dat \
                --output results/synthetic_search.csv \
                --window-sizes 500,1000,2000,5000 \
                --thresholds 0.5,1.0,2.0,4.0 \
                --frames 30
            '');
          };

          # Live comparison of all detectors (synthetic data)
          compare_live = {
            type = "app";
            program = toString (pkgs.writeShellScript "compare_live" ''
              ${envSetup}
              exec cargo run --release --bin compare_live -- \
                --config config/detectors.toml \
                data/synthetic/fan_test.dat
            '');
          };

          # Live comparison with real fan data
          compare_real = {
            type = "app";
            program = toString (pkgs.writeShellScript "compare_real" ''
              ${envSetup}
              exec cargo run --release --bin compare_live -- \
                --config config/detectors.toml \
                data/fan/fan_const_rpm.dat
            '');
          };

          # Main visualizer (pass data file as argument)
          visualize = {
            type = "app";
            program = toString (pkgs.writeShellScript "visualize" ''
              ${envSetup}
              exec cargo run --release -- "''${1:-data/fan/fan_const_rpm.dat}"
            '');
          };

          # Quick single-dataset evaluation
          evaluate_quick = {
            type = "app";
            program = toString (pkgs.writeShellScript "evaluate_quick" ''
              ${envSetup}
              cargo run --release --bin evaluate_cmax_slam -- \
                --data data/synthetic/fan_test.dat \
                --output results/cmax_quick.csv
            '');
          };

          # Full parameter sweep (generates datasets then evaluates)
          evaluate_sweep = {
            type = "app";
            program = toString (pkgs.writeShellScript "evaluate_sweep" ''
              ${envSetup}

              echo "Generating test datasets..."
              for rpm in 600 900 1200 1500 2000; do
                for blades in 2 3 4 5; do
                  cargo run --release --bin generate_synthetic -- \
                    --rpm $rpm --blades $blades \
                    --output "data/synthetic/sweep_rpm''${rpm}_b''${blades}.dat"
                done
              done

              echo "Running evaluation..."
              rm -f results/cmax_sweep.csv
              for f in data/synthetic/sweep_*.dat; do
                cargo run --release --bin evaluate_cmax_slam -- \
                  --data "$f" \
                  --output results/cmax_sweep.csv
              done

              echo "Results written to results/cmax_sweep.csv"
            '');
          };
        });
    };
}
