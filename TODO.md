
# ORDEN DEL DIA

- implement and verify the script for running tournaments with multiple models:
    - test random vs random (works). 
        - report any inconsistencies
        - collect 2-3 games and manually verify them
    - test random vs models
        - ensure model loading is correct
        - check result performance and ensure its right
        - collect 2-3 games and manually verify them
        - measure execution time for estimating how much it takes to run simulations
    - test models vs models
        - two different models with difference in accuracies
        - check result performance and ensure its right
        - measure execution time
    - one it works as-is, refactor the todos in the script

- 10 experiments from the ablations are ready! so they can be scp's here and analyzed, pull them and start making a script to run checkpoints against holdout, extract data and and produce plots / tables 
    - dk_128 are on their way too

- rsync polling script for pulling results from ablations trough ssh (need ssh key!!)

- subir datos a huggingface
Otros problemas:
- Las curvas de value head parecen planas. puede que haya un desequilibrio entre las pérdidas
- Variabilidad alta en biases de value head
- Cómo "leo" los histogramas de gradiente para saber si las capas profundas los propagan correctamente? 

-  plantearse registrar alguna otra métrica de performance a parte de accuracy... (f1, precision, recall, precision@k, recall@k, map@50, etc)

- sobre el test estadístico: hay q ejecutar los checlkpoints contra el holdout de test. pero para el "torneo"... cómo hago esto? uso una sola semilla? porque son 6 modelos entotal: baseline random, baseline ffn, input simple y cabezas simples, input simple y cabezas factorizadas, input 2d y cabezas simples, input 2d y cabezas factorizadas. 6*6 = 36 runs... 

- verify statistical significance and hypotheses: figure out if more runs with different random seeds are needed, and what statistical test is best for proving significance
    - find out the cause of the error in the 3090. it dont support torch.compile. it was some illegal memory access in CUDA, meybe related to fused adamw

- add .agents file and simple skills

- mejora de performance: añadir cabeza categórica de valor. ver si mejora. igual quitar dropout y subir grad clipping...
    - the derivative of $\tanh(x)$ is $1 - \tanh^2(x)$. As the prediction approaches $+1.0$ or $-1.0$, the derivative approaches $0$. If the model is highly confident but wrong (e.g., predicting $+0.99$ for a position that actually ends in a loss $-1.0$), the gradient drops to near-zero, making it extremely difficult for the optimizer to correct the error.
    - A categorical Value Head (KataGo / modern Leela): Instead of predicting a single scalar float, the model outputs logits for a discrete distribution over game outcomes (e.g., 3 classes: [Loss, Draw, Win]). This completely avoids regression saturation and yields much more stable gradients. This might be added as auxiliary though
    - Since $p_{\text{loss}}, p_{\text{draw}}, p_{\text{win}}$ (where $p_{\text{loss}} + p_{\text{draw}} + p_{\text{win}} = 1.0$), this can be mapped to continuous values via $$V = (-1.0 \times p_{\text{loss}}) + (0.0 \times p_{\text{draw}}) + (1.0 \times p_{\text{win}}) = p_{\text{win}} - p_{\text{loss}}$$

- updates in the docs:
  - add info about the dataset. annex holds figures with stats (escribir en la docu sobre el dataset. dar un sample. estadisticas en el anexo. explicar profundidades)
  - explain the test holdout strategy and why it's important to separate games using plies.  
  - update the transformer architecture
  - explain separately the backbone of the transformer and the head(s). 
  - explain the encoding of the inputs for each model (mlp and transformer w/ and w/out inductive bias)

- tune lr and other adam hyperparams on highest batchsize possible on a 3090. refer to google tuning playbook. do for different d_k and depths. stick to it 

- while this is going, put up some tests:
    - for dataloaders (?)
    - for parsing, and parsing working in different modes (mlp, transformer, transformer with inductive bias). for 
    - for forward passes of the models going right
    - for training dry runs and stuff being correctly saved: tensorboard, directories, .pt files, test evaluation...
    - for decode_move_indices and uci_to_index

# ideas no tan urgentes

- maybe... using some LR scheduler (warmup + decay) would be nice

-  ver si mejora el tiempo hacer non_blocking los tensores: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
- idea! puede ser que flashattn no sea tan util por lo pequeño de la secuencia, y como se sacrifica precision en bfloat16, mejor usar float32 con efficientAttention o otro backend? refs:
    - https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
    - https://docs.pytorch.org/docs/2.12/generated/torch.nn.attention.sdpa_kernel.html

- dudas transformer:
    - que inicialización? se puede usar la por defecto? parece xavier glorot en pytorch 
    https://github.com/pytorch/pytorch/blob/4f4b931aba66ae438aae8daca1dcbebeabb947e4/torch/nn/modules/activation.py#L1018-L1034
    - MUY IMPORTANTE! averiguar si el -1/0/1 es relativo al jugador actual o no!!! ese si lo pes, pero y la evaluación???
    - IMPORTANTE! decidir si hacer flipping de tablero (en dataloader y durante rollouts) o si añadir 2 tokens mas. si hago flip en RL, luego tengo que invertir los resultados para el negro
    - more residuals never hurt! figure out where they might be needed
- unit test the dataset parser! if it's not tested we cant trust it

- pensar si compensa tunear dinamicamente el peso de cada perdida, ya que creo que a la hora de buscar las mejores jugadas la value orienta mejor (homocedastic whatever loss)



---
---

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

## 2605 
- total investigation in docs
- formal model of problem + theoretical transformer architecture
- complete docs in tex

## 3105
- acabar aqruitectura transformer (dummy pass)
- escribir el loader apropiado para el nuevo transformer

## 0206

- refactorizar y separar modelo sde entrenamiento: una carpeta para cada
    - modelo em modelos/
    - training/ tiene los scripts +  utilidades como profiler / estimacion de tiempo
- averiguar el por que del colapso total de pesos a partir de d_k=128 y 256... vino despues de flash attention, igual es float16?
- flash attention y otras cosas
- entrenamientos / profilings varios para testear batch size / d_k / lr

- se que quiero que el modelo vaya mucho más rapido. y se que para eso necesito un batch size grande. así que por que no simplemente fijo el que quiero y hago busqueda del mejor lr para un par de modelos y ya? entonces, con batch size fijo de 4096, hacer optimizacion de rl para:
    - dk 64 y dept 4
    - dk 64 y dept 8
    - dk 256 y depth 4
    - dk 64 y depth 8 
    - tengo que pensar el num de iteraciones, y encontrar un bayesiano que funcione y se integre bien con el código

- averiguar que configuracion tiene el mejor equilibrio performance - tiempo de entrenamiento. entiende batch size vs learning rate y optimiza. es mas utilizacion de gpu = menos tiempo? igual no... (?)

## 0306

- escribir a leandro sobre gpu server

## 14-16/06
now i need to:
- tomarse en serio split train-val-test, hacer uno de test
- fix some arquitecture recipes (big and small) and commit to them for now
- fix hiperparams for them (lr, batch size, dtype and sdpa kernel)
- just go ahead and train them fully, like 50 epochs for example (al entrenar: probar con todo mezclado VS. con primero depth 2 e incrementar)

while they train:

- design coirrectly the inductive bias i mentioned for the input and output
  - choose one and see how to use it
  - see what needs to change in the code

- train reproducibly and cleanly:
    - first DECIDE ON THE DATA SPLIT USED! meaning: d2, d3 or d4 or merged? use this throughout the whole pipeline
    - then SPLIT IN TRAIN/TEST!!!! hold test out and use for later proofs.
    - small mlp simple encoding, big mlp simple encoding, small standard encoding, big standard encoding, small 2d + factored head encoding, big 2d + factored head encoding (consult previous results in mlp since its been long ago). pick adequate lr, stick with normal batch size and dtype
    - compare, draw conclusions. prove statistical significance

- ^did the above, but bat. changing 2 independent varibales makes finding results and conclusions impossible. ablate one by one!!

## 1706
- ran the exps for the ablation thing

- add scalar loss to the model (gradient norm, l2?) and gradient histogram (less frequent since it adds overhead, maybe in all of the first 5 epochs and then every 5?). remember to detach or call cpu

- also log head activaation outputs

- run the new ablatable experiments on the 3090. 

-  refactor a bunch of stuff from train

## 18-19/06
- correctly develop the experiment "procedure": what i do, starting from data and going trough the steps in processing, creating the model, choosing params, training it, testing experiments... i havent written it in the docs tough
- get the README ready for making the repo public.
- test the pyffish python api works as expected
- run hyperparam optim on 3090 searching for best lr and optim params for the biggest batch size i can on 3090, in order to speed up training
- on the other 3090: run ablation experiments for the 4 versions of the supervised transformer: simple input & simple heads, simple input & factored heads, 2d input & simple heads, 2d input & factored heads. use same hyperparams for all to ensure fair comparison and no extra independent variables.

- input simple y cabezas simples, input simple y cabezas factorizadas, input 2d y cabezas simples, input 2d y cabezas factorizadas
- escribir a david con experimentos hechos y pedir consejo sobre ablaciones. igual mejor opensourcear repo para q lo vea todo