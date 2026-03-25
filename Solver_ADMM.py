import numpy as np
import scipy.linalg as la

np.random.seed(42)

def admm_solver_rho(D, Phi, q, rho_0 = 10, max_iter = 50000, tol_abs = 1e-4, tol_rel = 1e-4,
                    tau_incr = 2, tau_decr = 2, mu = 10, maximize = True):
    
    '''
    Implementación del algoritmo ADMM para resolver el problema propuesto en la Sección 3.2
    Resuelve el problema:
        min  q^T y + I_C(z)
        s.a. D y - z = 0
    '''
    
    # Cargamos las dimensiones del problema. Pondremos los nombres en mayúsculas
    # para no liar la notación con la usada en el TF. Pero cabe notar que 
    # P = v + n + 1     y      N = n + 1
    
    P, N = D.shape
    
    # Hallamos la matriz D^t D, para tenerla ya cargada y no tenerla que calcular
    # varias veces, mejorando considerablemente el tiempo de computación
    
    DtD = D.T @ D
    
    # Para garantizar que la matriz sea def-pos, definimos
    
    holg = 1e-6 * np.eye(N)
    
    
    # Siguiendo las ideas de Boyd, vamos a factorizar esta matriz usando Cholesky
    # para reducir la complejidad computacional. Nótese que esto fallará si por 
    # fallo numérico la matriz DtD no es definida-positiva.
    
    try:
        L = la.cholesky(DtD + holg, lower = True)        
    except la.LinAlgError:
        raise ValueError("La matriz debe ser definida positiva.")
        
    # Inicialización de las variables
    
    # Variable y, pesos de la cartera:   
    y = np.zeros(N) 
    
    # Variables z, verificación del no-arbitraje
    
    z = np.zeros(P) 
    
    # Variable dual del problema escalado
    
    u = np.zeros(P) 
    
    # Bucle ADMM hasta un máximo de max_iter
    
    rho = rho_0
    
    if maximize:
        fun = lambda x: np.maximum(Phi, x)
        
    else: 
        fun = lambda x: np.minimum(Phi, x)
        q = - q
    
    
    for k in range(max_iter):
        
        # Almacenamos la solución de z del paso anterior para el residuo dual (s)
        
        z_1 = z.copy()
        
        # Empezamos a resolver nuestras variables. En primer lugar, la variable y,
        # que resolveremos usando la descomposición de Cholesky:
        
        lado_derecho = D.T @ (z - u) - q / rho
            
        y_intermedia = la.solve_triangular(L, lado_derecho, lower=True)
        
        y = la.solve_triangular(L.T, y_intermedia, lower=False)
        
        # A continuación, resolvemos z
        
        Dy = D @ y
        
        z = fun(Dy + u)
        
        # Hallamos el residuo primal, r
        
        r = Dy - z 
        
        # Acumulamos este error en la variable dual escalada u
        
        u += r 
        
        # Calculamos el residuo dual
        
        s = rho * D.T @ (z - z_1)
        
        # Para las condiciones de parada del algoritmo, hallamos las variables
        
        eps_pri = np.sqrt(P) * tol_abs + tol_rel * max(np.linalg.norm(Dy), np.linalg.norm(z))
        
        eps_dual = np.sqrt(N) * tol_abs + tol_rel * np.linalg.norm(rho * D.T @ u)
        
        norma_r = np.linalg.norm(r)
        
        norma_s = np.linalg.norm(s)
        
        if (norma_r < eps_pri) and (norma_s < eps_dual):
            
            print(f'Convergencia en la iteración {k}')
            
            if not maximize:
                q = - q
            
            return q @ y, y,  k, norma_r, norma_s
        
        # Si el método no ha convergido, hallamos la nueva rho
        
        if norma_r > mu * norma_s:
            rho = tau_incr * rho
            u = u / tau_incr
        elif norma_s > mu * norma_r:
            rho = rho / tau_decr
            u = u * tau_decr
        else:
            pass
        
    print("El método no convergió")
    print(f"La norma del residuo primal fue {norma_r}")
    print(f"La norma del residuo dual fue {norma_s}")
    
    if not maximize:
        q = - q    
    
    return q @ y, y, k, norma_r, norma_s


'''
    Definimos el número de activo y los respectivos pay-offs de cada uno de ellos
'''

n_activos = 1

S0, sigma, T = 100, 0.25, 1

K = 100

# Durante el trabajo asumimos, por simplicidad r=0

r = 0

# Número de escenarios. Hay que definir eps y beta a gusto 

eps = 0.01

beta = 0.01

N_escenarios = int((n_activos + 1) / (eps * beta) - 1)

'''
    Vamos a definir una función que nos dé aleatorios normales correlacionados
    usando una cópula normal.
'''

def genera_aleatorios_correl(n_activos, N_escenarios, correl=None):
  # Nos aseguramos de que los números sean del tipo int
    n_activos = int(n_activos)
    N_escenarios = int(N_escenarios)

  # Si solo hay un activo, simplemente lanzamos aleatorios
    if n_activos == 1:
        return np.random.normal(0, 1, N_escenarios)
    
    else:
  # Si hay varios activos, usamos la Id en caso de no tener matriz de correlaciones
        if correl is None:
            correl = np.eye(n_activos) 
        else:
            correl = np.array(correl)

      # Usamos la matriz de Cholesky para correlacionar los aleatorios
        L = la.cholesky(correl, lower = True)
        return L @ np.random.normal(0, 1, (n_activos, N_escenarios))
        

# Como son bajo el mismo subyacente, la correlación es la identidad y se puede
# generar un único vector de aleatorios

correl = np.array(1)

# Aleatorios 1 D

Z = genera_aleatorios_correl(n_activos, N_escenarios, correl)


# Creamos el vector de subyacentes

ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


# Strikes del Teorema 3.5

strikes = np.array([80, 90, 100, 110, 120])

# Precios de las opciones con dichos strikes

precios = np.array([22.5, 14.0, 7.5, 3.5, 1.0])

# Strike objetivo

K = 105

# El vector de precios q es directamente las primas observadas en mercado

q = np.concatenate(([1], precios))

# Inicializamos la matriz D
D = np.ones((N_escenarios, len(strikes)+1))

# Las columnas son el payoff de la opción correspondiente en cada escenario

for i, strike in enumerate(strikes):
    D[:, i+1] = np.maximum(ST - strike, 0)
    
# Phi es el payoff de la opción que queremos acotar
Phi = np.maximum(ST - K, 0)    

# Función que nos devuelve las cotas analíticas según el Teorema 3.5

def cotas_strike(precios, strikes, K):
    
    m_sup = (precios[3] - precios[2]) / (strikes[3] - strikes[2])
    pi_sup = precios[2] + m_sup * (K - strikes[2])
    
    # Cota Inferior
    m_izq = (precios[2] - precios[1]) / (strikes[2] - strikes[1])
    pi_inf_izq = precios[2] + m_izq * (K - strikes[2])
    
    m_der = (precios[4] - precios[3]) / (strikes[4] - strikes[3])
    pi_inf_der = precios[3] + m_der * (K - strikes[3])
    
    pi_inf = max(pi_inf_izq, pi_inf_der)
        
    return pi_inf, pi_sup

# Cotas analíticas

pi_inf, pi_sup = cotas_strike(precios, strikes, K)

print(f"Cota Superior: {pi_sup:.4f} €")
print(f"Cota Inferior: {pi_inf:.4f} €")

# Cota Superior ADMM
admm_sup_price, weights_sup, _,_,_ = admm_solver_rho(D, Phi, q, maximize=True, rho_0=10)

# Cota Inferior ADMM
admm_inf_price, weights_inf, _,_,_ = admm_solver_rho(D, Phi, q, maximize=False, rho_0=10)

print(f"Cota Superior: {admm_sup_price:.4f} €")
print(f"Cota Inferior: {admm_inf_price:.4f} €")


    
    
    
    
    
    
    
    
