{ pkgs ? (import <nixpkgs> {}), ... }:
pkgs.mkShell {
  packages = with pkgs; [
    (python3.withPackages (ps: [
      ps.jedi-language-server
      ps.loguru
      ps.matplotlib
      ps.opencv4
    ]))
  ];
}
