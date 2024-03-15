import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def polinomio(x):
    # Se define el polinomio aquí, en este caso: f(x) = x^3 - 2x - 5
    return x**3 - 2*x - 5

def derivada_polinomio(x):
    # Se define la derivada del polinomio aquí, en este caso: f'(x) = 3x^2 - 2
    return 3*x**2 - 2

def metodo_newton(polinomio, derivada_polinomio, x0, tolerancia=1e-6, max_iter=100):
    iteracion = 0
    raices = []

    while iteracion < max_iter:
        fx = polinomio(x0)
        # Se guardan las raices en el arreglo
        raices.append(x0)

        if abs(fx) < tolerancia:
            break

        f_derivada_x = derivada_polinomio(x0)

        if f_derivada_x == 0:
            print("Derivada igual a cero. No se puede continuar.")
            return None
        x0 = x0 - fx / f_derivada_x
        iteracion += 1

    print(raices)

    # Graficar el polinomio
    valor_x = np.linspace(-3, 3, 100)
    valor_y = polinomio(valor_x)
    plt.plot(valor_x, valor_y, label="f(x) = x^3 - 2x - 5", color="blue")

    # Graficar las raíces encontradas sobre el polinomio
    raices_y = polinomio(np.array(raices))
    plt.scatter(raices, raices_y, color="red", s=10, label="Raíces")

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title("Método de Newton-Raphson")
    plt.legend()
    plt.grid(True)
    plt.show()


# Especifica el punto inicial x0
x0 = 2.0

# Llama al método de Newton con la función y su derivada
metodo_newton(polinomio, derivada_polinomio, x0)