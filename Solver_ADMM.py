import numpy as np
import scipy.linalg as la

np.random.seed(42)

def admm_solver_rho(D, Phi, q, rho_0 = 10, max_iter = 50000, tol_abs = 1e-4, tol_rel = 1e-4,
                    tau_incr = 2, tau_decr = 2, mu = 10, maximize = True):
    
    '''
    Implementación del algoritmo ADMM para resolver el problema propuesto en la Sección 3.2
    Resuelve el problema:
        min  q^T x + I_C(z)
        s.a. D x - z = 0
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
