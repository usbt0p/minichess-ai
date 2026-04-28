
- posiblemente sea buena idea hacer que la red supervisada prediga tammbien la score? sigue los targets del paper 

- vamos a hacer caso a karpathy. revisar los datos primero; comprobar duplicados, unir los distintos datasets y ver cuantos tengo realmente
    - convertir a texto
    - parser del dataset en binario
    - añadir score al parser
    - analizar edge cases de los datos: ver cuantas promotions hay y como están distribuidas
    para saber si es importante representarlas (modificar archivo c++)
    - si los datos no son suficientes... probar a generar más y en el peor de los casos, investigar alguna alternativa como a) destilar otro modelo más grande b) probar contrastive learning c) resignarse a puro RL

- el curriculum learning se puede hacer contra otros modelos "decentes" como por ejemplo full ataque, o una ponderacion simple de (mate, jaque, captura, desarrollo)

- desacoplar dataset de la funcion de training
- abstraer cosas repetitivas del trainign pipeline
- asegurar una validación correcta sin contaminación
- añadir test split al dataloader

