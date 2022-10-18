build:
    clear
    nix build -L --cores 4

run_random: build
    result/bin/Posit16GMRES -im test_data/random40.mtx -iv test_data/random40.vec -iters 100 -o /mnt/RamDisk/random -hh
    cat /mnt/RamDisk/random.csv

run_sherman: build
    result/bin/Posit64QMR -im test_data/mat.mtx -iv test_data/vec.vec -iters 200 -hh -restart 30 -sparse -o /mnt/RamDisk/sherman -precond
    cat /mnt/RamDisk/sherman.csv
