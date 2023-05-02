# vkr_project

## Общее описание

Данный проект представляет собой реализацию численного решения задачи сложного теплообмена в системе теплопроводящих стержней квадратного сечения. При этом преполагается, что теплообмен с внешней средой, а также между стержнями происходит по закону излучения Стефана-Больцмана для абсолютно черного тела.

\begin{cases}
	- \lambda \Delta u = f, (x,y) \in G \\
	\lambda \dfrac{\partial u}{\partial n} + \sigma |u||u^3| = g, (x,y) \in \partial G.
\end{cases}


