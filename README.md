# Kernel_methods_course

This repository consists of the practical sessions of Kernel Methods in Machine Learning course taught by [Prof. Jean-Philippe Vert](https://jpvert.github.io/) and T.A. [Juliette Marrie](https://www.linkedin.com/in/juliette-marrie-5b8a59179/?originalSubdomain=fr) in July 2023, [AMMI 2023 program](https://aimsammi.org/).

------------

In [practical session 2](https://github.com/omerahmed12345elhussien/Kernel_methods_course/blob/main/Practical_Session_2.ipynb), we implement Ridge Regression, Weighted Ridge Regression, and Logistic Ridge Regression.
### Ridge Regression 

Given $X \in \mathbb{R}^{n \times p}$ and $y \in \mathbb{R}^n$, solve
$$\min_{\beta \in \mathbb{R}^p} \frac{1}{n} \|\|y - X \beta\|\|^2 + \lambda \|\|\beta\|\|^2 .$$

### Weighted Ridge Regression 

Given $X \in \mathbb{R}^{n \times p}$ and $y \in \mathbb{R}^n$, and weights $w \in \mathbb{R}^n_+$, solve
$$\min_{\beta \in \mathbb{R}^p} \frac{1}{n} \sum_{i=1}^n w_i (y_i - \beta^\top x_i)^2 + \lambda \|\|\beta\|\|^2 .$$

### Logistic Ridge Regression

Given $X \in \mathbb{R}^{n \times p}$ and $y \in \lbrace -1,+1\rbrace ^n$, solve
$$\min_{\beta \in \mathbb{R}^p} \frac{1}{n} \sum_{i=1}^n \log (1+e^{-y_i \beta^\top x_i}) + \lambda \|\|\beta\|\|^2 .$$

In [practical session 3](https://github.com/omerahmed12345elhussien/Kernel_methods_course/blob/main/Practical_Session_3.ipynb), we implement Hard margin & Soft margin Support Vector Machines using Quadratic Programming.

solves the quadratic program

$$\begin{aligned}
\min_x &  \frac{1}{2}x^\top P x + q^\top x \\
\mathrm{s.t. }  & Gx \leq h \\
& Ax = b
\end{aligned}$$

- $P, q$ define the objective
- $G, h$ are all the inequality constraints
- $A, b$ are all the equality constraints

### Primal problem (hard margin) <a name="hard-margin-primal"></a>
$$\begin{aligned}
\min_{w, b} &  \frac{1}{2}w^\top w \\
\mathrm{s.t. }  &\hspace{0.2cm} y_i x_i^\top w + y_i b \geq 1  ,\hspace{0.2cm} \forall i \in [1, n]\\
\end{aligned}$$

### Dual (hard margin): <a name="hard-margin-dual"></a>
$$\begin{aligned}
\max_\alpha & \hspace{0.2cm} \mathrm{1}^\top\alpha - \frac{1}{2}\alpha^\top X_y X_y^\top \alpha \\
\mathrm{s.t. }  &\hspace{0.2cm} \alpha \geq 0 \\
& \hspace{0.2cm} y^\top\alpha = 0
\\
\end{aligned}$$

### Primal problem (soft margin): <a name="soft-margin-primal"></a>
$$\begin{aligned}
\min_{w, b, \xi} & \hspace{0.2cm} \frac{1}{2}w^\top w + C \mathbf{1}^\top \xi \\
\mathrm{s.t. } \hspace{0.2cm} & \xi \geq 0 \\
& y_i x_i^\top w + y_i b + \xi_i\geq 1
\\
\end{aligned}$$

### Dual (soft margin): <a name="soft-margin-dual"></a>
$$\begin{aligned}
\max_\alpha & \hspace{0.2cm} \mathrm{1}^\top\alpha -\frac{1}{2}\alpha^\top X_y X_y^T \alpha \\
\mathrm{s.t. } \hspace{0.2cm} & \alpha \geq 0 \\
& \alpha \leq C \\
& y^\top\alpha = 0
\\
\end{aligned}$$

In [practical session 4](https://github.com/omerahmed12345elhussien/Kernel_methods_course/blob/main/Practical_Session_4_.ipynb), we implement the Radial Basis Function (RBF) kernel, the median heuristic approach, and Kernel Ridge Regression.

The Radial Basis Function (RBF) kernel with parameter $\sigma$ is defined as follows:
$$k_\sigma(x, y) = \exp \left( -\frac{\|\|x-y\|\|^2}{2 \sigma^2}\right)$$

The median heuristic approach is given by:
$$\sigma \approx \mathrm{median} \lbrace\|\|x_i-x_j\|\|:i,j=1,\dots, n\rbrace.$$
### Kernel Ridge Regression
The prediction rule is given by:
$$\hat{y}(X_t) = K_{X_t, X} (K_{X, X} + \lambda n I)^{-1} y$$



In [practical session 5](https://github.com/omerahmed12345elhussien/Kernel_methods_course/blob/main/Practical_Session_5_.ipynb), we implement Kernel Logistic Regression, and Kernel SVM.
### Kernel Logistic Regression

 Optimization problem:

 The kernel version of the logistic regression problem is stated as follows:
$$\min_\alpha J(\alpha) =  -\frac{1}{n} \sum_{i=1}^n {\log \hspace{0.2cm} \sigma(y_i (K \mathbf{\alpha})_i) } + \lambda \mathbf{\alpha}^\top K \mathbf{\alpha} .$$

Decision function:

The decision function is then $$\hat{f}(X_{test}) = \sigma(K_{X_{test}, X_{train}} \alpha)$$
  
### Kernel SVM
Soft-margin dual problem:

$$\begin{aligned}
\max_\alpha & \hspace{0.2cm} \mathrm{1}^\top\alpha -\frac{1}{2}\alpha^\top y_{diag} X X^\top y_{diag} \alpha \\
\mathrm{s.t. } \hspace{0.2cm}& \alpha \geq 0 \\
& \alpha \leq C \\
& y^\top\alpha = 0
\\
\end{aligned}$$

Decision function:

$$f(X_{test}) = X_{test}X_{train}^\top \beta + b$$
 where:
  - $\beta := \alpha \odot y$ is the elementwise product of $\alpha$ and $y$ ($\beta_i = \alpha_i y_i$)
  - $b$ is the average of $X_{sv} X_{train}^\top \beta$
  
  $sv$ = support vectors = subset of training vectors





