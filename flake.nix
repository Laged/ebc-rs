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
          combine (with stable; [ clippy rustc cargo rustfmt rust-src ]);
      };

      devShells = forEachSupportedSystem ({ pkgs }:
        let
          commonDeps = with pkgs; [ rustToolchain pkg-config ];

          linuxDeps = with pkgs; [
            alsa-lib
            udev
            libxkbcommon
            wayland
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
    };
}
