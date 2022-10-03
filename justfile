build:
    clear
    nix build -L

run_random: build
    result/bin/LongDoubleGMRES -im ../Evaluations/result/inputs/inputs/random40.mtx -iv ../Evaluations/result/inputs/inputs/random40.vec -iters 100 -o /mnt/RamDisk -hh -sparse