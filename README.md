# 📈 Cotas de No-Arbitraje de Opciones Europeas con ADMM

Repositorio con el código fuente y la memoria del trabajo final de la asignatura sobre Optimización Convexa en Ciencia de Datos (Máster en Matemáticas Avanzadas, UNED).

## 🎯 Resumen del Proyecto
Este proyecto implementa un *solver* personalizado basado en el método ADMM (Alternating Direction Method of Multipliers) para calcular cotas de no arbitraje en mercados financieros. 

Frente a la clásica relajación por Programación Semidefinida (SDP) de Bertsimas y Popescu, este algoritmo:
- **Garantiza la coherencia financiera**, exigiendo precios positivos.
- **Escala en alta dimensión**, resolviendo un problema de momentos generalizados mediante la discretización probabilística de Calafiore y Campi.
- **Acelera el cálculo**, explotando la geometría del problema y pre-calculando la factorización de Cholesky sobre matrices estáticas.

## 🚀 Uso del Solver
Se adjunta el solver programado junto con un ejemplo de ejecución (el Teorema 3.5 de la memoria, que nos da las cotas analíticas en caso de conocer otras opciones con el mismo vencimiento sobre el mismo activo básico).

## 📄 Memoria Completa
Para profundizar en la formulación matemática (dualidad, condición de Slater y garantías probabilísticas), puedes consultar el documento adjunto: 'Uso del ADMM para las cotas de no arbitraje de opciones europeas.pdf'.
