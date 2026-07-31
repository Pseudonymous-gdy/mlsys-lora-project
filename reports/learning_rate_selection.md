Common-grid validation sweep (8 cells, 24 runs, seeds per cell: [3])

| Method | 2e-5 | 5e-5 | 1e-4 | 2e-4 |
|---|---|---|---|---|
| Full FT | **39.9 ± 2.2** | 21.5 ± 3.1 | 6.6 ± 1.4 | 1.1 ± 0.4 |
| LoRA-r16 | 39.7 ± 2.4 | 43.5 ± 2.8 | 45.9 ± 0.6 | **46.5 ± 1.5** |

Per-rate detail

| Method | AdamW learning rate | Validation first-turn EM (%) | Validation loss | Seeds |
|---|---|---|---|---|
| Full FT | 2e-5 | **39.9 ± 2.2** | 0.4210 | 11, 22, 33 |
| Full FT | 5e-5 | 21.5 ± 3.1 | 0.5078 | 11, 22, 33 |
| Full FT | 1e-4 | 6.6 ± 1.4 | 0.6630 | 11, 22, 33 |
| Full FT | 2e-4 | 1.1 ± 0.4 | 0.9868 | 11, 22, 33 |
| LoRA-r16 | 2e-5 | 39.7 ± 2.4 | 0.5076 | 11, 22, 33 |
| LoRA-r16 | 5e-5 | 43.5 ± 2.8 | 0.4841 | 11, 22, 33 |
| LoRA-r16 | 1e-4 | 45.9 ± 0.6 | 0.4697 | 11, 22, 33 |
| LoRA-r16 | 2e-4 | **46.5 ± 1.5** | 0.4565 | 11, 22, 33 |

Selected learning rates

  Full FT: 2e-5 (39.9 ± 2.2 %)
  LoRA-r16: 2e-4 (46.5 ± 1.5 %)

LaTeX

\begin{tabular}{lcccc}
\toprule
& $2\times10^{-5}$ & $5\times10^{-5}$ & $10^{-4}$ & $2\times10^{-4}$ \\
\midrule
Full FT & $\mathbf{39.9 \pm 2.2}$ & $21.5 \pm 3.1$ & $6.6 \pm 1.4$ & $1.1 \pm 0.4$ \\
LoRA-r16 & $39.7 \pm 2.4$ & $43.5 \pm 2.8$ & $45.9 \pm 0.6$ & $\mathbf{46.5 \pm 1.5}$ \\
\bottomrule
\end{tabular}
