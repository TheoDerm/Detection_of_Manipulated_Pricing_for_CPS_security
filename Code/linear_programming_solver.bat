:: This script creates a new directory named: "Graphs" to store the created graphs
:: and uses the lp_solve software to solve the .lp files and save the results to a new directory named solved_lp
if not exist Graphs mkdir Graphs
move lp_solve.exe linear_prog_files
cd linear_prog_files
if not exist solved_lp mkdir solved_lp
for %%k in (*.lp) do (lp_solve %%k > solved_lp\%%~nk.txt)
move lp_solve.exe ..



