import numpy as np
import matplotlib.pyplot as plt

"""
Given dataset
"""
x=np.array([-1.67245526, -2.36540279, -2.14724263,  1.40539096,  1.24297767,
        -1.71043904,  2.31579097,  2.40479939, -2.22112823])
y=np.array([-18.56122168, -24.99658931, -24.41907817,  -2.688209  ,
        -1.54725306,  -19.18190097,   1.74117419,
         3.97703338, -24.80977847])

"""
For each value of `n`, we specify the best parameters which were calculated experimentally.
"""
parameters = {
    "1": {"gamma": 1e-1},
    "2": {"gamma": 1e-2},
    "3": {"gamma": 1e-2},
    "4": {"gamma": 1e-3},
    "5": {"gamma": 1e-4},
}
    
"""
Calculate f(x) for all x in X given the weights $w$.
"""
def calculate_f_x(X, w):
  output = []
  for x in X:
    out = 0
    for i in range(len(w)):
      out += (x**i)*w[i]
      
    output.append(out)
  return output

"""
Derivatives. All derivatives are also multiplied by `calculate_f_x` or `err`. 
To reduce duplicate code, we multiple this within the gradient_descent loop.
"""
derivatives = [
  lambda x: 2,
  lambda x: 2*x,
  lambda x: 2*(x**2),
  lambda x: 2*(x**3),
  lambda x: 2*(x**4),
  lambda x: 2*(x**5),
]

def gradient_descent(n, gamma, x, y, threshold):
  """
  Initialize weights to something small.
  """
  w = np.random.uniform(-1e-3, 1e-3, n+1)

  """
  Training metrics
  """
  num_iterations = 0
  total_err = []
  total_err.append(sum(((calculate_f_x(x, w) - y)**2) / len(x)))

  """
  Training loop, run until change in error drops below a `threshold`
  """
  while True:
    num_iterations += 1

    """
    Calculate the change in $w$ using all training samples.
    """
    w_delta = np.zeros(n+1)
    for i in range(len(x)):
      err = calculate_f_x([x[i]], w) - y[i]
      for n_ in range(n+1):
        w_delta[n_] += (err * derivatives[n_](x[i]))
    w_delta /= len(x)  

    """
    Update $w$ and record the error after the update.
    """
    w -= (w_delta * gamma)
    total_err.append(sum(((calculate_f_x(x, w) - y)**2) / len(x)))

    """
    If change in error is less than `threshold`, we can stop the loop.
    """
    if len(total_err) > 1 and abs(total_err[-2] - total_err[-1]) < threshold:
      break

  return total_err, w, num_iterations

if __name__ == "__main__":
  ns = range(1,6)
  mae = []
  total_iterations = []

  """
  Train for all values of $n$
  """
  for n in ns:
      """
      Train and calculate metrics
      """
      gamma = parameters[str(n)]['gamma']
      err_list, w, num_iterations = gradient_descent(n, gamma, x, y, threshold=1e-5)
      print(f"n={n}, mae={err_list[-1]}, num_iterations={num_iterations}")
      mae.append(err_list[-1])
      total_iterations.append(num_iterations)

      """
      Plot samples, predicted values for $n$. Save to File
      """
      x_ = np.arange(-3,3,0.1)
      y_ = np.array(calculate_f_x(x_,w))
      plt.plot(x_, y_, ':k')
      plt.plot(x, calculate_f_x(x,w), '.r')
      plt.plot(x, y, '.b')
      plt.legend(["Prediction Function", "Prediction Per Sample", "Sample"])
      plt.savefig(f"n.{n}.eps")
      plt.clf()
    
  """
  Plot Mean Squared Error per $n$. Save to File
  """
  plt.plot(ns, mae, '*-k')
  plt.xlabel("n")
  plt.ylabel("MSE")
  plt.savefig(f"mse_per_n.eps")
  plt.clf()

  """
  Plot Total number of iterations per $n$. Save to File
  """
  plt.plot(ns, total_iterations, '*-k')
  plt.xlabel("n")
  plt.ylabel("Total Iterations")
  plt.savefig(f"iterations_per_n.eps")
  plt.clf()
