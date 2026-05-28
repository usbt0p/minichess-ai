
- posiblemente sea buena idea hacer que la red supervisada prediga tammbien la score? sigue los targets del paper 

- datos en uso: puedo usar más seguramente. no setoy usando los plys, ni el turno, y puede que haya mas

- el curriculum learning se puede hacer contra otros modelos "decentes" como por ejemplo full ataque, o una ponderacion simple de (mate, jaque, captura, desarrollo)

- abstraer cosas repetitivas del trainign pipeline
- asegurar una validación correcta sin contaminación
- añadir test split al dataloader

# cambios en el stats.cpp

- vamos a hacer caso a karpathy. revisar los datos primero; comprobar duplicados, unir los distintos datasets y ver cuantos tengo realmente
    - parser del dataset en binario
    - analizar edge cases de los datos: ver cuantas promotions hay y como están distribuidas
    para saber si es importante representarlas (modificar archivo c++)
    - si los datos no son suficientes... probar a generar más y en el peor de los casos, investigar alguna alternativa como a) destilar otro modelo más grande b) probar contrastive learning c) resignarse a puro RL


---


# ORDEN DEL DIA
- acabar aqruitectura transformer (dummy pass)
- dudas:
    - que inicialización? se puede usar la por defecto? parece xavier glorot en pytorch 
    https://github.com/pytorch/pytorch/blob/4f4b931aba66ae438aae8daca1dcbebeabb947e4/torch/nn/modules/activation.py#L1018-L1034
- unit test the dataset parser! if it's not tested we cant trust it

# DONE

## 2904
1. hacer script para ver mejoras de escalado con los datos
2. unir datasets. comprobar duplicados
3. hacer lo de los cambios en stats.cpp
4. plottear estadísticas. cambiar el plot para que añada numero exacto al histograma
- desacoplar dataset de la funcion de training
- sacar el máximo y mínimo valor de score, y hacer 30 bins entre min y max (centrados en 0) para ver qué sale y si se puede usar int16 para representarlos.
    para esto hay que cambiar stats.cpp

## 3004
- plottear overlapped los resultados para el dataset full merged, el full d2
    - averiguar por que la mejora no es tanta como esperaba. 
    - probar com d4 benchamark
    - probar a darle más epochs / un modelo mas grande al full merged
    - sacar conclusiones...
- mientras hago todo lo demas, dejar un benchmark del simple pero con dropout + batchnorm e igual una residual connection?
- crear un baseline random que solo elija aleatoriamente movimientos y evaluaciones, para comparar con el mio

## 0505
- visualizar un par de tableros aleatorios? los top endgames de los stats.txt. 
-  crear una nueva baseline, que tenga encoding para coronación y masking de movs ilegales
    - pensar encoding de coronacion
    - IMPORTANTE!!!: implementar el masking!!! no va a ir bien la prediccion si no se penaliza lo suficiente a los movimientos ilegales


## 0605
- hacer el benchmark para la baseline con masking y coronacion y añadir al overlapped
