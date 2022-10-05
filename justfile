build:
    clear
    nix build -L

run_random: build
    result/bin/DoubleQMR -im test_data/random40.mtx -iv test_data/random40.vec -iters 100 -o /mnt/RamDisk/random -hh -precond
    cat /mnt/RamDisk/random.csv

run_sherman: build
    result/bin/DoubleQMR -im test_data/mat.mtx -iv test_data/vec.vec -iters 4000 -hh -restart 30 -sparse -o /mnt/RamDisk/sherman -precond
    cat /mnt/RamDisk/sherman.csv