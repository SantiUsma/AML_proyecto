# AML_proyecto

Para reproducir los resultados debe correr el siguiente comando:

>- python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg [archivo a la configuración] --output [carpeta con los pesos entrenados] --data-path [carpeta con la base de datos] --local_rank [numero de gpu] --mode test --fold [numero del fold a evaluar] --adv [True si se esta evaluando en el dataset de ejemplos adversarios]

El archivo de configuración es: "configs/swinv2/swinv2_tiny_patch4_window8_256.yaml"
La carpeta de output donde se encuentran los pesos son: "/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/output_stablelike2_fold1/" o "/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/output_stablelike2_fold2/" dependiendo del fold a evaluar.
La carpeta de la base de datos son: "/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_fold1_original/" o "/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_fold2_original/" dependiendo del fold a evaluar. 

Para sacar la predicción de una unica imagen correr el siguiente comando:

>- python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg [archivo a la configuración] --output [carpeta con los pesos entrenados] --data-path [carpeta con la base de datos] --local_rank [numero de gpu] --mode demo --img [ruta a una imagen de prueba]

Un ejemplo de imagen de prueba se encuentra en: "/media/SSD3/asusma/DELFOS2/DELFOS/Swin-Transformer/dataset_fold1/train/Negative/1013580545.2.4C.jpeg"

Los pesos y la base de datos se encuentran en el servidor Lambda002.
