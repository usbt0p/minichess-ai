
- posiblemente sea buena idea hacer que la red supervisada prediga tammbien la score? sigue los targets del paper 

- datos en uso: puedo usar más seguramente. no setoy usando los plys, ni el turno, y puede que haya mas

- IMPORTANTE!!!: implementar el masking!!! no va a ir bien la prediccion si no se penaliza lo suficiente a los movimientos ilegales

- el curriculum learning se puede hacer contra otros modelos "decentes" como por ejemplo full ataque, o una ponderacion simple de (mate, jaque, captura, desarrollo)

- desacoplar dataset de la funcion de training
- abstraer cosas repetitivas del trainign pipeline
- asegurar una validación correcta sin contaminación
- añadir test split al dataloader

# cambios en el stats.cpp

- vamos a hacer caso a karpathy. revisar los datos primero; comprobar duplicados, unir los distintos datasets y ver cuantos tengo realmente
    - parser del dataset en binario
    - analizar edge cases de los datos: ver cuantas promotions hay y como están distribuidas
    para saber si es importante representarlas (modificar archivo c++)
    - si los datos no son suficientes... probar a generar más y en el peor de los casos, investigar alguna alternativa como a) destilar otro modelo más grande b) probar contrastive learning c) resignarse a puro RL

- sacar el máximo y mínimo valor de score, y hacer 40 bins entre min y max para ver qué sale y si se puede usar int16 para representarlos

- visualizar un par de tableros aleatorios?

# ORDEN DEL DIA
1. hacer script para ver mejoras de escalado con los datos
2. unir datasets. comprobar duplicados
3. hacer lo de los cambios en stats.cpp
4. plottear estadísticas. cambiar el plot para que añada numero exacto al histograma
5. crear una nueva baseline, que tenga encoding para coronación y masking de movs ilegales
